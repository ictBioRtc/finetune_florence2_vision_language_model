# modules/utils.py


"""
Shared utilities for Florence-2 modules
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging
from pathlib import Path
import json
from transformers import AutoProcessor, AutoModelForCausalLM
from threading import Thread
import time
from huggingface_hub import login
import io

# Configure logging
def setup_logging(log_dir='logs'):
    """Set up logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'florence2_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('florence2'), log_file

# Define colormap for visualizations
COLORMAP = [
    'red', 'blue', 'green', 'orange', 'purple', 
    'brown', 'pink', 'gray', 'olive', 'cyan'
]

# Task prompts mapping
TASK_PROMPTS = {
    "Object Detection": "<OD>",
    "Caption": "<CAPTION>",
    "Detailed Caption": "<DETAILED_CAPTION>",
    "More Detailed Caption": "<MORE_DETAILED_CAPTION>"
}

class ModelManager:
    """Manager class for loading and caching models"""
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.logger, _ = setup_logging()
        
    def get_model(self, model_id, device=None):
        """Get model and processor, loading if necessary"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        # Check if already loaded
        cache_key = f"{model_id}_{device}"
        if cache_key in self.models:
            return self.models[cache_key], self.processors[cache_key]
            
        # Load model and processor
        self.logger.info(f"Loading model: {model_id} on {device}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32
            ).eval().to(device)
            
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Cache for future use
            self.models[cache_key] = model
            self.processors[cache_key] = processor
            
            self.logger.info(f"Successfully loaded model: {model_id}")
            return model, processor
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise

    def authenticate_hub(self, token):
        """Authenticate with Hugging Face Hub"""
        try:
            login(token=token)
            self.logger.info("Successfully authenticated with Hugging Face Hub")
            return True
        except Exception as e:
            self.logger.error(f"Failed to authenticate: {str(e)}")
            return False

# Global model manager instance
model_manager = ModelManager()

def stream_logs(file_path, output_fn):
    """Stream log file to output function"""
    def _stream_logs():
        # Wait for file to be created
        while not os.path.exists(file_path):
            time.sleep(0.5)
            
        with open(file_path, 'r') as f:
            # Go to the end of file
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    output_fn(line)
                else:
                    time.sleep(0.1)
    
    log_thread = Thread(target=_stream_logs)
    log_thread.daemon = True
    log_thread.start()
    return log_thread

class ProgressMonitor:
    """Helper class to monitor and update progress in UI"""
    def __init__(self, progress=None, status=None):
        self.progress = progress
        self.status = status
        self.current = 0
        self.total = 1
        self.status_text = ""
        
    def update(self, current, total, status_text):
        """Update progress and status text"""
        self.current = current
        self.total = total
        self.status_text = status_text
        
        if self.progress is not None:
            try:
                # For Gradio progress objects
                self.progress(value=current/total, desc=status_text)
            except Exception as e:
                # Fallback if progress isn't a Gradio progress object
                pass
                
        if self.status is not None:
            try:
                self.status(status_text)
            except Exception:
                pass
        
        return current / total, status_text

def visualize_object_detection(image, prediction, image_format='pil'):
    """Visualize object detection results"""
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for i, (bbox, label) in enumerate(zip(prediction['bboxes'], prediction['labels'])):
        # Get a consistent color for this instance
        color_idx = i % len(COLORMAP)
        color = COLORMAP[color_idx]
        
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )

        # Add the rectangle to the Axes
        ax.add_patch(rect)

        # Annotate with label
        plt.text(
            x1,
            y1-5,
            label,
            color='white',
            fontsize=12,
            bbox=dict(facecolor=color, alpha=0.7)
        )

    # Remove the axis ticks and labels
    ax.axis('off')
    
    # Add title with detection count
    plt.title(f"Detected {len(prediction['labels'])} objects", fontsize=16)

    # Return as appropriate format
    if image_format == 'pil':
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    else:
        plt.tight_layout()
        return fig

def visualize_caption(image, caption, image_format='pil'):
    """Visualize image with generated caption"""
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(image)
    
    # Remove the axis ticks and labels
    ax.axis('off')
    
    # Add the caption as text below the image, not as title
    # This ensures it's visible and doesn't get cut off
    caption_text = caption if isinstance(caption, str) else str(caption)
    
    # Use a text box at the bottom of the image for better visibility
    ax.text(0.5, -0.1, caption_text, 
            wrap=True,
            fontsize=12,
            ha='center',
            va='top',
            transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Adjust the figure to make room for the caption
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add extra space at bottom for caption
    
    # Return as appropriate format
    if image_format == 'pil':
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    else:
        return fig