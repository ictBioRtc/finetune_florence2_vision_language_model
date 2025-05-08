#!/usr/bin/env python3
"""
Florence-2 Gradio App

A modular Gradio interface for working with Microsoft's Florence-2 model:
- Inference tab: Run inference with the original Microsoft model
- Training tab: Fine-tune the model on custom datasets
- Evaluation tab: Test fine-tuned models
"""

import os
import torch
import gradio as gr
from PIL import Image
import logging
from pathlib import Path
import tempfile
import time
import shutil
from threading import Thread

# Import our modules
from modules import inference, training, evaluation, utils

# Set up logging
logger, log_file = utils.setup_logging()

# Model choices
MODEL_CHOICES = [
    "microsoft/Florence-2-base-ft",
    "microsoft/Florence-2-large-ft"
]

# Task choices
TASK_CHOICES = ["Caption", "Detailed Caption", "More Detailed Caption"]

def inference_fn(image, task, model_id):
    """Run inference with Florence-2 model"""
    if image is None:
        return None, "Please upload an image."
    
    # Run inference
    try:
        results, inf_time = inference.run_inference(image, task, model_id)
        processed_result = inference.process_result(image, results, task)
        
        # Format message based on task
        if task == "Object Detection":
            task_prompt = utils.TASK_PROMPTS[task]
            detection = results.get(task_prompt, {})
            labels = detection.get('labels', [])
            message = f"Detected {len(labels)} objects in {inf_time:.2f} seconds: {', '.join(labels)}"
        else:
            task_prompt = utils.TASK_PROMPTS[task]
            caption = results.get(task_prompt, "")
            message = f"Generated caption in {inf_time:.2f} seconds"

        return processed_result, message
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}")
        return None, f"Error: {str(e)}"

def train_fn(dataset_name, base_model, epochs, batch_size, learning_rate, freeze_vision, progress=gr.Progress()):
    """Fine-tune Florence-2 model on custom dataset"""
    try:
        # Create output directory
        output_dir = f"outputs/finetuned_{int(time.time())}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Progress monitor for UI updates
        # The issue is likely here - progress.tqdm might not be what you expect
        monitor = utils.ProgressMonitor(
            progress=progress,  # Just pass the progress object directly
            status=None
        )
        
        # Training configuration
        config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'save_frequency': 1,
            'num_workers': 2,
            'freeze_vision': freeze_vision
        }
        
        # Start training
        model_path = training.train_model(
            dataset_name=dataset_name,
            base_model_id=base_model,
            output_dir=output_dir,
            config=config,
            progress_monitor=monitor
        )
        
        return model_path, f"Training completed! Model saved to {model_path}"
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        return None, f"Error: {str(e)}"

def eval_inference_fn(image, task, model_source, model_path, trained_model_path):
    """Run inference with fine-tuned Florence-2 model"""
    if image is None:
        return None, "Please upload an image."
    
    # Determine which model to use
    if model_source == "Use trained model from current session":
        if not trained_model_path:
            return None, "No trained model available. Please train a model first or select 'Load model from path/Hub'."
        model_to_use = trained_model_path
    else:
        if not model_path:
            return None, "Please provide a model path or Hub ID."
        model_to_use = model_path
    
    # Use the same inference code but with the fine-tuned model
    try:
        results, inf_time = inference.run_inference(image, task, model_to_use)
        processed_result = inference.process_result(image, results, task)
        
        # Format message based on task
        if task == "Object Detection":
            task_prompt = utils.TASK_PROMPTS[task]
            detection = results.get(task_prompt, {})
            labels = detection.get('labels', [])
            message = f"Detected {len(labels)} objects in {inf_time:.2f} seconds: {', '.join(labels)}"
        else:
            task_prompt = utils.TASK_PROMPTS[task]
            caption = results.get(task_prompt, "")
            message = f"Generated caption in {inf_time:.2f} seconds"

        return processed_result, message
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}")
        return None, f"Error: {str(e)}"

def upload_to_hub_fn(model_path, repo_id, token):
    """Upload model to Hugging Face Hub"""
    try:
        # Authenticate with Hub
        success = utils.model_manager.authenticate_hub(token)
        if not success:
            return "Failed to authenticate with Hugging Face Hub. Please check your token."
        
        # Import necessary functions
        from huggingface_hub import create_repo, upload_folder
        
        # First, create the repository if it doesn't exist
        try:
            logger.info(f"Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                token=token,
                repo_type="model",
                exist_ok=True  # This will not raise an error if the repo already exists
            )
            logger.info(f"Repository created or already exists: {repo_id}")
        except Exception as e:
            logger.error(f"Error creating repository: {str(e)}")
            return f"Error creating repository: {str(e)}"
        
        # Then, push to Hub
        logger.info(f"Uploading model from {model_path} to {repo_id}")
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token  # Explicitly pass token here
        )
        
        return f"Successfully uploaded model to {repo_id}"
    except Exception as e:
        logger.error(f"Error uploading to Hub: {str(e)}")
        return f"Error: {str(e)}"

def create_app():
    """Create Gradio app"""
    with gr.Blocks(title="Florence-2 Interface") as app:
        gr.Markdown("# Florence-2 Model Interface")
        gr.Markdown("A unified interface for running inference, fine-tuning, and evaluating Microsoft's Florence-2 vision model")
        
        with gr.Tabs():
            # Inference Tab
            # Inference Tab with Memory Management
            with gr.Tab("Inference"):
                gr.Markdown("## Run Inference with Florence-2")
                
                with gr.Row():
                    with gr.Column():
                        inf_image = gr.Image(type="pil", label="Upload Image")
                        inf_task = gr.Dropdown(choices=TASK_CHOICES, value=TASK_CHOICES[0], label="Task")
                        inf_model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Model")
                        
                        # Add a clear cache button
                        inf_clear_cache_btn = gr.Button("Clear GPU Memory")
                        inf_btn = gr.Button("Run Inference")
                    
                    with gr.Column():
                        inf_output = gr.Image(label="Results")
                        inf_message = gr.Textbox(label="Status")
                
                # Memory cleanup function
                def clear_gpu_memory_inf():
                    """Clear GPU memory cache"""
                    try:
                        # Clear PyTorch cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # Force garbage collection
                            import gc
                            gc.collect()
                            # Clear model manager cache if possible
                            if hasattr(utils.model_manager, 'models'):
                                utils.model_manager.models.clear()
                            if hasattr(utils.model_manager, 'processors'):
                                utils.model_manager.processors.clear()
                            return "GPU memory cleared successfully!"
                        else:
                            return "No GPU available."
                    except Exception as e:
                        logger.error(f"Error clearing GPU memory: {str(e)}")
                        return f"Error clearing memory: {str(e)}"
                
                # Memory-optimized inference function
                def memory_optimized_inference_fn(image, task, model_id):
                    """Run inference with Florence-2 model with memory optimization"""
                    if image is None:
                        return None, "Please upload an image."
                    
                    # Run inference
                    try:
                        results, inf_time = inference.run_inference(image, task, model_id)
                        processed_result = inference.process_result(image, results, task)
                        
                        # Format message based on task
                        if task == "Object Detection":
                            task_prompt = utils.TASK_PROMPTS[task]
                            detection = results.get(task_prompt, {})
                            labels = detection.get('labels', [])
                            message = f"Detected {len(labels)} objects in {inf_time:.2f} seconds: {', '.join(labels)}"
                        else:
                            task_prompt = utils.TASK_PROMPTS[task]
                            caption = results.get(task_prompt, "")
                            message = f"Generated caption in {inf_time:.2f} seconds"
                        
                        # Clear memory after inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        return processed_result, message
                    except Exception as e:
                        logger.error(f"Error in inference: {str(e)}")
                        # Clear memory even if there's an error
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return None, f"Error: {str(e)}"
                
                # Connect the clear button
                inf_clear_cache_btn.click(
                    fn=clear_gpu_memory_inf,
                    inputs=[],
                    outputs=[inf_message]
                )
                
                # Connect the inference button
                inf_btn.click(
                    fn=memory_optimized_inference_fn,
                    inputs=[inf_image, inf_task, inf_model],
                    outputs=[inf_output, inf_message]
                )
            
            # Training Tab
            with gr.Tab("Training"):
                gr.Markdown("## Fine-tune Florence-2 on Custom Dataset")
                
                with gr.Row():
                    with gr.Column():
                        train_dataset = gr.Textbox(label="Dataset Name (HF Hub)", placeholder="username/dataset-name")
                        train_model = gr.Dropdown(choices=MODEL_CHOICES, value=MODEL_CHOICES[0], label="Base Model")
                        train_epochs = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Epochs")
                        train_batch_size = gr.Slider(minimum=1, maximum=16, value=4, step=1, label="Batch Size")
                        train_lr = gr.Slider(minimum=1e-6, maximum=1e-4, value=2e-6, step=1e-7, label="Learning Rate")
                        train_freeze = gr.Checkbox(value=True, label="Freeze Vision Tower")
                        train_btn = gr.Button("Start Training")
                    
                    with gr.Column():
                        train_output = gr.Textbox(label="Trained Model Path")
                        train_message = gr.Textbox(label="Status")
                
                train_btn.click(
                    fn=train_fn,
                    inputs=[train_dataset, train_model, train_epochs, 
                            train_batch_size, train_lr, train_freeze],
                    outputs=[train_output, train_message]
                )
                
                with gr.Accordion("Upload to Hub", open=False):
                    hub_model_path = gr.Textbox(
                        label="Model Path",
                        info="Local path to the model you want to upload"
                    )
                    hub_repo_id = gr.Textbox(
                        label="Repository ID",
                        placeholder="username/model-name",
                        info="Your Hugging Face username followed by a model name (e.g., johndoe/florence2-finetuned)"
                    )
                    hub_token = gr.Textbox(
                        label="Hugging Face Token",
                        type="password",
                        info="Get your token from huggingface.co/settings/tokens"
                    )
                    hub_btn = gr.Button("Upload to Hub")
                    hub_status = gr.Textbox(label="Upload Status")
                    
                    hub_btn.click(
                        fn=upload_to_hub_fn,
                        inputs=[hub_model_path, hub_repo_id, hub_token],
                        outputs=hub_status
                    )
            
            # Evaluation Tab - Simplified to just run inference on fine-tuned model
            # Evaluation Tab with Memory Management
            with gr.Tab("Evaluation"):
                gr.Markdown("## Test Fine-tuned Model")
                
                with gr.Row():
                    with gr.Column():
                        eval_image = gr.Image(type="pil", label="Upload Image")
                        eval_task = gr.Dropdown(choices=TASK_CHOICES, value=TASK_CHOICES[0], label="Task")
                        
                        # Always require a model path
                        eval_model_path = gr.Textbox(
                            label="Model Path or Hub ID",
                            placeholder="path/to/model or username/model-name",
                            info="Enter local path to model or Hugging Face model ID"
                        )
                        
                        # Add a clear cache button for memory management
                        clear_cache_btn = gr.Button("Clear GPU Memory")
                        eval_btn = gr.Button("Run Inference")
                    
                    with gr.Column():
                        eval_output = gr.Image(label="Results")
                        eval_message = gr.Textbox(label="Status")
                
                # Memory cleanup function
                def clear_gpu_memory():
                    """Clear GPU memory cache"""
                    try:
                        # Clear PyTorch cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # Force garbage collection
                            import gc
                            gc.collect()
                            # Clear model manager cache if possible
                            if hasattr(utils.model_manager, 'models'):
                                utils.model_manager.models.clear()
                            if hasattr(utils.model_manager, 'processors'):
                                utils.model_manager.processors.clear()
                            return "GPU memory cleared successfully!"
                        else:
                            return "No GPU available."
                    except Exception as e:
                        logger.error(f"Error clearing GPU memory: {str(e)}")
                        return f"Error clearing memory: {str(e)}"
                
                # Memory-optimized inference function
                def memory_optimized_inference(image, task, model_path):
                    """Run inference with specified model using memory optimization"""
                    if image is None:
                        return None, "Please upload an image."
                    
                    if not model_path:
                        return None, "Please provide a model path or Hub ID."
                    
                    # Use the same inference code with the specified model
                    try:
                        # Run inference
                        results, inf_time = inference.run_inference(image, task, model_path)
                        processed_result = inference.process_result(image, results, task)
                        
                        # Format message based on task
                        if task == "Object Detection":
                            task_prompt = utils.TASK_PROMPTS[task]
                            detection = results.get(task_prompt, {})
                            labels = detection.get('labels', [])
                            message = f"Detected {len(labels)} objects in {inf_time:.2f} seconds: {', '.join(labels)}"
                        else:
                            task_prompt = utils.TASK_PROMPTS[task]
                            caption = results.get(task_prompt, "")
                            message = f"Generated caption in {inf_time:.2f} seconds"
                        
                        # Clear memory after inference
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        return processed_result, message
                    except Exception as e:
                        logger.error(f"Error in inference: {str(e)}")
                        # Make sure to clear memory even if there's an error
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return None, f"Error: {str(e)}"
                
                # Connect the clear button
                clear_cache_btn.click(
                    fn=clear_gpu_memory,
                    inputs=[],
                    outputs=[eval_message]
                )
                
                # Connect the evaluate button
                eval_btn.click(
                    fn=memory_optimized_inference,
                    inputs=[eval_image, eval_task, eval_model_path],
                    outputs=[eval_output, eval_message]
                )
        # Event handlers to copy paths between tabs
        train_output.change(
            fn=lambda x: x,
            inputs=train_output,
            outputs=hub_model_path
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)
