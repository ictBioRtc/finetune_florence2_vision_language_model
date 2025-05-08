# modules/training.py

"""
Florence-2 training module
"""
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
# New import (the fix)
from torch.optim import AdamW  # Import from torch.optim instead
from transformers import get_scheduler
from datasets import load_dataset, concatenate_datasets
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Union
import time
from tqdm.auto import tqdm
import torch.cuda.amp  # For automatic mixed precision
from torch.amp import autocast  # Modern import for autocast

from .utils import model_manager, setup_logging, ProgressMonitor

logger, log_file = setup_logging()

class MultiTaskDataset(Dataset):
    """Dataset class for Florence-2 training"""
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get task-specific prompt
        prompt = example['prompt']
                   
        # Get image
        image = example['image']
        if hasattr(image, 'mode') and image.mode != "RGB":
            image = image.convert("RGB")
            
        return prompt, example['label'], image

def collate_fn(batch, processor):
    """
    Generic collate function for both tasks
    """
    prompts, targets, images = zip(*batch)
    
    # Get image sizes for later use in post-processing
    image_sizes = [(img.width, img.height) for img in images]
    
    inputs = processor(
        text=list(prompts),
        images=list(images),
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding='max_length'
    )
    
    # Store image sizes for later post-processing
    inputs["image_sizes"] = image_sizes
        
    return inputs, targets

def load_sharded_dataset(dataset_name: str) -> Dict[str, Dataset]:
    """
    Load and combine sharded datasets
    """
    # Load all splits from the hub
    all_splits = load_dataset(dataset_name)
    
    # Handle training shards
    train_splits = []
    if "train" in all_splits:
        train_splits.append(all_splits["train"])
    for split_name in all_splits.keys():
        if split_name.startswith("train_shard_"):
            train_splits.append(all_splits[split_name])
            
    if len(train_splits) == 0:
        raise ValueError("No training splits found!")
    train_full = concatenate_datasets(train_splits)
    logger.info(f"Training dataset loaded with {len(train_full)} examples.")
    
    # For validation, only load the "val" split as no validation shards exist.
    if "val" not in all_splits:
        raise ValueError("No validation splits found!")
    val_full = all_splits["val"]
    logger.info(f"Validation dataset loaded with {len(val_full)} examples.")
    
    return {
        "train": train_full,
        "val": val_full
    }

class TrainingMonitor:
    """Monitor and visualize training progress"""
    def __init__(self, save_dir: str = "training_metrics"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.lr_history = []
        self.best_val_loss = float('inf')
        
    def update(self, train_loss: float, val_loss: float, current_lr: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.lr_history.append(current_lr)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False
        
    def plot_metrics(self, epoch: int):
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(self.lr_history)
        plt.title('Learning Rate vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_metrics_epoch_{epoch}.png')
        plt.close()
        
        return str(self.save_dir / f'training_metrics_epoch_{epoch}.png')

# Fix 1: Update the training loop in evaluate_model function
def evaluate_model(model, processor, val_loader, device):
    """Evaluate loss on validation set"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            # Get model's data type
            dtype = next(model.parameters()).dtype
            
            # Move tensors to device with proper type conversion
            input_ids = inputs["input_ids"].to(device)  # Keep as Long/Int
            
            # Apply gradient scaling for fp16 if needed
            if dtype == torch.float16:
                # Ensure pixel values don't overflow in fp16
                pixel_values = inputs["pixel_values"].to(device, dtype=torch.float32)
                pixel_values = pixel_values.to(dtype)  # Safely convert after moving to device
            else:
                pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
                
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)  # Keep as is
                
            # Tokenize targets
            labels = processor.tokenizer(
                text=list(targets),
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(device)
            
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Check for NaN loss - add error handling
            if torch.isnan(outputs.loss):
                print("WARNING: NaN loss detected in validation. Skipping batch.")
                continue
                
            val_loss += outputs.loss.item()
    
    # Handle case where all losses are NaN
    if len(val_loader) == 0:
        avg_val_loss = float('nan')
    else:
        avg_val_loss = val_loss / len(val_loader)
        
    logger.info(f"Validation Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss

# Fix 2: Update the main training loop with gradient scaling for FP16
def train_model(
    dataset_name: str,
    base_model_id: str,
    output_dir: str,
    config: Dict,
    progress_monitor: Optional[ProgressMonitor] = None,
) -> str:
    """
    Train Florence-2 model on custom dataset
    
    Args:
        dataset_name: Hugging Face dataset name
        base_model_id: Base model ID
        output_dir: Output directory
        config: Training configuration
        progress_monitor: Optional progress monitor for UI updates
        
    Returns:
        Path to fine-tuned model
    """
    # Setup
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('learning_rate', 2e-5)
    epochs = config.get('epochs', 5)
    save_frequency = config.get('save_frequency', 1)
    num_workers = config.get('num_workers', 2)
    freeze_vision = config.get('freeze_vision', True)
    
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Update progress
    if progress_monitor:
        progress_monitor.update(0, 1, "Loading dataset...")
    
    # Load dataset
    dataset = load_sharded_dataset(dataset_name)
    
    # Create datasets
    train_dataset = MultiTaskDataset(dataset['train'])
    val_dataset = MultiTaskDataset(dataset['val'])
    
    # Update progress
    if progress_monitor:
        progress_monitor.update(0.1, 1, "Loading model...")
    
    # Initialize model and processor
    model, processor = model_manager.get_model(base_model_id, device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=num_workers,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, processor),
        num_workers=num_workers
    )
    
    # Optional: Freeze vision tower
    if freeze_vision:
        logger.info("Freezing vision tower parameters")
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    
    # Initialize training components
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # IMPORTANT: Disable mixed precision - run in full precision to avoid errors
    torch.set_default_dtype(torch.float32)
    if hasattr(model, 'to'):
        # Convert model back to float32 if it was in float16
        model = model.to(torch.float32)
    
    # Initialize monitor
    monitor = TrainingMonitor(save_dir / 'metrics')
    
    # Update progress
    if progress_monitor:
        progress_monitor.update(0.2, 1, "Starting training...")
    
    logger.info("Starting training")
    logger.info(f"Training on device: {device}")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    logger.info(f"Number of training examples: {len(train_loader.dataset)}")
    logger.info(f"Number of validation examples: {len(val_loader.dataset)}")
    
    # Main training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        valid_batches = 0
        
        # Calculate epoch progress percentage
        epoch_start_progress = 0.2 + (epoch / epochs) * 0.7
        epoch_end_progress = 0.2 + ((epoch + 1) / epochs) * 0.7
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            
            # Move tensors to device
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            # Tokenize targets
            labels = processor.tokenizer(
                text=list(targets),
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False
            ).input_ids.to(device)
            
            # Forward pass (without AMP)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected in batch {batch_idx}. Skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Track loss
            train_loss += loss.item()
            valid_batches += 1
            
            # Update progress
            if progress_monitor:
                batch_progress = batch_idx / len(train_loader)
                current_progress = epoch_start_progress + batch_progress * (epoch_end_progress - epoch_start_progress)
                progress_monitor.update(
                    current_progress, 
                    1, 
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )
        
        # Calculate average loss
        avg_train_loss = train_loss / max(1, valid_batches)
        
        # Validation phase
        if progress_monitor:
            progress_monitor.update(epoch_end_progress, 1, f"Epoch {epoch+1}/{epochs}, Evaluating...")
            
        model.eval()
        val_loss = 0
        valid_val_batches = 0
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc="Evaluating"):
                inputs, targets = val_batch
                
                # Move tensors to device
                input_ids = inputs["input_ids"].to(device)
                pixel_values = inputs["pixel_values"].to(device)
                
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    
                # Tokenize targets
                labels = processor.tokenizer(
                    text=list(targets),
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False
                ).input_ids.to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Skip if NaN
                if torch.isnan(outputs.loss):
                    continue
                    
                val_loss += outputs.loss.item()
                valid_val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, valid_val_batches)
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Update training monitor
        is_best = monitor.update(
            avg_train_loss,
            avg_val_loss,
            optimizer.param_groups[0]['lr']
        )
        metrics_plot = monitor.plot_metrics(epoch + 1)
        
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Save checkpoints
        if is_best and not torch.isnan(torch.tensor(avg_val_loss)):
            logger.info(f"New best validation loss: {avg_val_loss:.4f}")
            model_save_path = save_dir / 'best_model'
            model.save_pretrained(model_save_path)
            processor.save_pretrained(model_save_path)
        
        if (epoch + 1) % save_frequency == 0:
            output_dir = save_dir / f'checkpoint_epoch_{epoch+1}'
            output_dir.mkdir(exist_ok=True)
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
    
    # Final save
    model_save_path = save_dir / 'final_model'
    model.save_pretrained(model_save_path)
    processor.save_pretrained(model_save_path)
    
    # Update progress
    if progress_monitor:
        progress_monitor.update(1, 1, "Training completed!")
    
    return str(model_save_path)