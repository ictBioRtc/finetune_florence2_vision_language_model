# Fine-tune Vision AI Model for Volume Recognition

This project demonstrates how to fine-tune a vision AI model for recognizing fluid volumes in test tubes, with applications across medical, laboratory, and industrial settings.

## Prerequisites

### 1. HuggingFace Setup (Required)
1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens
3. Create a new token (read access)
4. Copy and save your token - you'll need it later

## Quick Start

1. Open terminal in your JarvisLabs workspace:
   ```bash
   File > New Launcher > Terminal
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/ictBioRtc/finetune_florence2_vision_language_model.git
   ```

3. Navigate to project directory:
   ```bash
   cd finetune_vision_ai_model
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Copy the public URL provided (e.g., https://ff20bc33e416f3319f.gradio.live)
7. Open in a new browser tab

## Using the Application

### Step 1: Test Initial Model (Inference Tab)
1. Unzip the provided `test_images.zip`
2. Go to "Inference" tab
3. Upload a test image
4. Leave other settings at default
5. Click "Run Inference"
6. Observe how the untrained model performs

### Step 2: Train the Model (Training Tab)
1. Dataset: `ictbiortc/beaker-volume-recognition-dataset`
2. Change epochs to 15 (for workshop purposes)
3. Click "Start Training"
4. Note: Full training could take ~5 hours

### Step 3: Upload Model to HuggingFace
1. After training completes, click "Upload to Hub"
2. Enter your model name (e.g., `your-username/beaker-volume-recognition-model`)
3. Paste your HuggingFace token
4. Click "Upload"

### Step 4: Important Configuration Update
1. Go to your model on HuggingFace
2. Navigate to "Files and versions"
3. Find `config.json`
4. Edit line 165 from:
   ```json
   "model_type": "",
   ```
   to:
   ```json
   "model_type": "davit",
   ```

### Step 5: Evaluate Your Model
1. Return to the app
2. Go to "Evaluate" tab
3. Upload a test image
4. Use your trained model
5. Compare results with the initial inference

## Applications

This volume recognition model has potential applications in:
- IV Fluid Monitoring
- Laboratory Automation
- Medication Dosing
- Urine Monitoring
- Manufacturing Quality Control
- Chemical Processing
- Beverage Industry
- Petroleum Industry

## Training Notes

- Full training typically takes days
- Workshop version uses 15 epochs (~5 hours)
- Larger epoch numbers yield better results
- GPU acceleration is recommended

## Troubleshooting

Common issues:
1. "Model not loading": Check your internet connection
2. "Training too slow": Verify GPU availability
3. "Upload failed": Verify your HuggingFace token
4. "Config error": Double-check the davit model_type update

## Next Steps

After successful training:
1. Experiment with different epochs
2. Try different image types
3. Test various fluid volumes
4. Integrate with your specific use case

Congratulations! You've successfully:
1. Tested a base vision model
2. Fine-tuned it for volume recognition
3. Uploaded it to HuggingFace
4. Created a practical AI solution for real-world applications

This workshop demonstrates how vision language models can be adapted for specific industrial and medical applications.
