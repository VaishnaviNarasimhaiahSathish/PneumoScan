# PneumoScan: Pneumonia Detection from Chest X-rays

PneumoScan is a deep learning project that classifies chest X-ray images as **Normal** or **Pneumonia** using a fine-tuned ResNet-18 model.  
The project contains the full pipeline:

- Data preprocessing  
- Model training  
- Evaluation  
- Saving the trained model  
- Interactive Streamlit web app for inference  

This repository is designed as an end-to-end example of how to build, train, and deploy a medical imaging classifier.

---

## Project Overview

Pneumonia is a common lung infection that can be detected through chest X-ray images.  
This project uses transfer learning on **ResNet-18** to perform binary classification on X-ray images.

The goal was to create a **simple, reliable, and easy-to-use** pipeline.

---

## 📁 Repository Structure
- data
    - train 
    - test 
    - val
- model 
    - pneumonia_resnet18_fast.pth ( Saved model weights)
- train.py (training setup)
- app.py (Streamlit Web App setup)
- requirements.txt
- README.md

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/PneumoScan.git
cd PneumoScan

2. Create the conda environment
conda create -n pneumo python=3.10
conda activate pneumo

3. Install the required packages
pip install -r requirements.txt

4. Training the Model
If you want to retrain or fine-tune the model:
python train.py

The script will:
1. Load images from data/train, data/val, data/test
2. Train a lightweight classifier head on ResNet-18
3. Save the best performing model under model/
Training defaults to 3 epochs for speed but can be increased

##Running the Web App

Launch the Streamlit app with:
streamlit run app.py

You can upload a chest X-ray image, and the model will output:
- Prediction: Normal or Pneumonia
- Confidence score
- A color-coded result box (green/red)

Model Performance (with a frozen backbone):

After 3 fast epochs:
- Validation Accuracy: ~75%
- Test Accuracy: ~82%

Performance improves with additional fine-tuning.

## Techniques Used

- Transfer Learning (ResNet-18)
- Image preprocessing & normalization
- Feature extraction (frozen backbone)
- PyTorch training loop
- Model checkpointing
- Streamlit UI for inference
- Simple CSS styling for cleaner UI
 
# Future Improvements

- Grad-CAM heatmaps for explainability
- Deployment on Streamlit Cloud

## App Preview

Below is an example of the PneumoScan web interface:

<p align="center">
  <img src="AppDemo/appdemo1" alt="App Screenshot" width="550">
</p>

