# modules/evaluation.py

"""
Florence-2 evaluation module
"""
import torch
from PIL import Image
import os
import logging
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Union
from pathlib import Path
import matplotlib.patches as patches
import numpy as np
import io

from .utils import model_manager, TASK_PROMPTS, COLORMAP, setup_logging

logger, _ = setup_logging()

def run_inference_for_eval(
    model,
    processor,
    image,
    task_prompt,
    device
) -> Dict[str, Any]:
    """Run inference for evaluation"""
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch.float16)
    # processed_inputs = processor(
    #     text=task_prompt,
    #     images=image,
    #     return_tensors="pt"
    # )
    
    # # Create a new dict for the processed inputs
    # inputs = {}
    # # Convert tensors selectively based on their role
    # for key, tensor in processed_inputs.items():
    #     if isinstance(tensor, torch.Tensor):
    #         if key == "pixel_values":
    #             # Convert image tensors to float16
    #             inputs[key] = tensor.to(device, torch.float16)
    #         else:
    #             # Keep other tensors (like input_ids) as their original type
    #             inputs[key] = tensor.to(device)
    #     else:
    #         inputs[key] = tensor
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
    
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=False
    )[0]
    
    # Clean and post-process
    generated_text = generated_text.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

def compare_models(
    original_model_id: str,
    finetuned_model_path: str,
    image: Image.Image,
    task_name: str
) -> Image.Image:
    """
    Compare original and fine-tuned models on the same image
    
    Args:
        original_model_id: Original model ID
        finetuned_model_path: Path to fine-tuned model
        image: Input image
        task_name: Task name
    
    Returns:
        Comparison visualization as PIL Image
    """
    # Get task prompt
    if task_name in TASK_PROMPTS:
        task_prompt = TASK_PROMPTS[task_name]
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load original model
    logger.info(f"Loading original model: {original_model_id}")
    original_model, original_processor = model_manager.get_model(original_model_id, device)
    
    # Load fine-tuned model
    logger.info(f"Loading fine-tuned model: {finetuned_model_path}")
    finetuned_model, finetuned_processor = model_manager.get_model(finetuned_model_path, device)
    
    # Run inference with both models
    logger.info("Running inference with original model")
    original_result = run_inference_for_eval(
        original_model, 
        original_processor, 
        image, 
        task_prompt, 
        device
    )
    
    logger.info("Running inference with fine-tuned model")
    finetuned_result = run_inference_for_eval(
        finetuned_model, 
        finetuned_processor, 
        image, 
        task_prompt, 
        device
    )
    
    # Visualize comparison based on task
    if task_name == "Object Detection":
        comparison_img = compare_od_results(
            image, 
            original_result.get(task_prompt, {}), 
            finetuned_result.get(task_prompt, {})
        )
    else:
        comparison_img = compare_caption_results(
            image,
            original_result.get(task_prompt, ""),
            finetuned_result.get(task_prompt, "")
        )
    
    return comparison_img

def compare_od_results(
    image: Image.Image, 
    original_result: Dict[str, Any], 
    finetuned_result: Dict[str, Any]
) -> Image.Image:
    """Compare and visualize object detection results"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original model results
    axes[0].imshow(image)
    axes[0].set_title("Original Model", fontsize=16)
    axes[0].axis('off')
    
    # Plot bounding boxes for original model
    for bbox, label in zip(original_result.get('bboxes', []), original_result.get('labels', [])):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, label, color='white', fontsize=10, 
                    bbox=dict(facecolor='red', alpha=0.7))
    
    # Plot fine-tuned model results
    axes[1].imshow(image)
    axes[1].set_title("Fine-tuned Model", fontsize=16)
    axes[1].axis('off')
    
    # Plot bounding boxes for fine-tuned model
    for bbox, label in zip(finetuned_result.get('bboxes', []), finetuned_result.get('labels', [])):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='g', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].text(x1, y1-5, label, color='white', fontsize=10, 
                   bbox=dict(facecolor='green', alpha=0.7))
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def compare_caption_results(
    image: Image.Image,
    original_caption: str,
    finetuned_caption: str
) -> Image.Image:
    """Compare and visualize caption results"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot original model results
    axes[0].imshow(image)
    axes[0].set_title("Original Model", fontsize=16)
    axes[0].axis('off')
    
    # Add caption as text box
    if original_caption:
        axes[0].text(0, -0.1, original_caption, wrap=True, 
                    transform=axes[0].transAxes, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot fine-tuned model results
    axes[1].imshow(image)
    axes[1].set_title("Fine-tuned Model", fontsize=16)
    axes[1].axis('off')
    
    # Add caption as text box
    if finetuned_caption:
        axes[1].text(0, -0.1, finetuned_caption, wrap=True, 
                   transform=axes[1].transAxes, fontsize=12,
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
