# modules/inference.py

"""
Florence-2 inference module
"""
import torch
from PIL import Image
import os
import logging
import time
from typing import Dict, Any, Tuple, Union
import io
from .utils import model_manager, visualize_object_detection, visualize_caption, TASK_PROMPTS, setup_logging

logger, _ = setup_logging()

def run_inference(
    image: Image.Image,
    task_name: str,
    model_id: str = "microsoft/Florence-2-base-ft",
    device: str = None
) -> Tuple[Dict[str, Any], float]:
    """
    Run inference with Florence-2 model
    
    Args:
        image: Input image
        task_name: Task name (must be in TASK_PROMPTS)
        model_id: Model ID or path
        device: Device to use (default: auto-detect)
        
    Returns:
        Tuple of (results, inference_time)
    """
    # Get task prompt
    if task_name in TASK_PROMPTS:
        task_prompt = TASK_PROMPTS[task_name]
    else:
        raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(TASK_PROMPTS.keys())}")
    
    # Get model and processor
    model, processor = model_manager.get_model(model_id, device)
    device = next(model.parameters()).device
    
    # Prepare inputs
    inputs = processor(
        text=task_prompt,
        images=image,
        return_tensors="pt"
    ).to(device, torch.float16)
    
    # Record inference time
    start_time = time.time()
    
    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.2f} seconds")
    
    # Decode the generated IDs
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]
    
    # Clean up generated text
    cleaned_text = generated_text.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
    
    # Post-process based on task
    parsed_result = processor.post_process_generation(
        cleaned_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    # Log the result for debugging
    logger.info(f"Parsed result: {parsed_result}")
    
    return parsed_result, inference_time

def process_result(
    image: Image.Image,
    result: Dict[str, Any],
    task_name: str
) -> Union[Image.Image, str]:
    """
    Process result for display
    
    Args:
        image: Original input image
        result: Inference result from run_inference
        task_name: Task name
        
    Returns:
        Processed result (image or text)
    """
    task_prompt = TASK_PROMPTS[task_name]
    
    if task_name == "Object Detection":
        # Extract detection results
        detection = result.get(task_prompt, {})
        return visualize_object_detection(image, detection)
    else:
        # Extract caption
        caption = result.get(task_prompt, "")
        
        # Log the extracted caption for debugging
        logger.info(f"Extracted caption for {task_name}: {caption}")
        
        # For caption tasks, always visualize the caption with the image
        # This ensures something visible is returned to Gradio
        return visualize_caption(image, caption)