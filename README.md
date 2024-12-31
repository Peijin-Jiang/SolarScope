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

### 3.Data Availability
The fine tuning and validation data required to run the models is from four studies:

1) Bradbury, K., Saboo, R., L. Johnson, T., Malof, J. M., Devarajan, A., Zhang, W., M. Collins, L., & G. Newell, R. (2016). Distributed solar photovoltaic array location and extent dataset for remote sensing object identification. Scientific Data, 3(1), 160106. https://doi.org/10.1038/sdata.2016.106
2) Khomiakov, M., Radzikowski, J. H., Schmidt, C. A., Sørensen, M. B., Andersen, M., Andersen, M. R., & Frellsen, J. (2022). SolarDK: A high-resolution urban solar panel image classification and localization dataset (arXiv:2212.01260). arXiv. https://doi.org/10.48550/arXiv.2212.01260
3) Kasmi, G., Saint-Drenan, Y.-M., Trebosc, D., Jolivet, R., Leloux, J., Sarr, B., & Dubus, L. (2023). A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata. Scientific Data, 10(1), 59. https://doi.org/10.1038/s41597-023-01951-4
4) Jiang, H., Yao, L., Lu, N., Qin, J., Liu, T., Liu, Y., & Zhou, C. (2021). Multi-resolution dataset for photovoltaic panel segmentation from satellite and aerial imagery. Earth System Science Data, 13(11), 5389–5401. https://doi.org/10.5194/essd-13-5389-2021


The prediction images for Kamakura, Japan can be obtained by running the prediction_image_import.ipynb notebook (with your own Google Staic Maps API)


After downloading the data, replace the paths in the Jupyter Notebooks to point to the corresponding local directories.
