# SolarScope: Estimating Urban Mining Potential of Distributed Solar Panels

## Project Overview
**SolarScope** is a computer vision (CV) and material flow analysis (MFA)-based model designed to estimate the future urban mining potential of distributed solar panel materials. This project integrates satellite imagery, deep learning techniques, and material assessment to predict and analyze the spatial distribution and material recycling of installed solar panels over time.

### Model Overview
The model architecture and workflow are illustrated below:

![Model Overview](Figure/Figure%201.png)

## Environment Setup
To replicate this project and run the provided Jupyter Notebooks, follow the steps below to set up the required environment.

### 1. Clone the Repository
git clone https://github.com/Peijin-Jiang/SolarScope
cd SolarScope

### 2. Install Dependencies
pip install -r requirements.txt

## 3.Data Availability
The data required to run the models is available on Google Drive:
Download Data Here: https://drive.google.com/drive/folders/1--VpnyFNceSNRUuWiRHE3j8LVSQ0-alL?usp=drive_link

Download and Replace Paths
After downloading the data, replace the paths in the Jupyter Notebooks to point to the corresponding local directories.

## Structure
solar_CV/
│
├── Figure/                  # Model overview and visualizations
├── Japan/                   # Prediction CSV
├── MFA/                     # Material Flow Analysis Data
├── src/                     # Source code for CV
├── dMFA.ipynb               # Dynamic MFA notebook
├── make_solar_pv_dataset.ipynb # CV Dataset preparation
├── panel_prediction.ipynb   # Solar panel prediction
├── prediction_image_import.ipynb # Load prediction images
├── solar_pv_classifier.ipynb # Classification model fine tuning
├── solar_pv_ctraining.ipynb # Segmentation model fine tuning
└── requirements.txt         # Environment dependencies

