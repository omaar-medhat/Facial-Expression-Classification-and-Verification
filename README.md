# Facial Expression Classification and Verification


This project implements two tasks related to facial recognition using deep learning techniques with TensorFlow and Keras. The codebase is provided in the `classification_and_verification.py` script, originally developed in a Jupyter Notebook (`Classification_and_Verification.ipynb`).

## Project Overview

### Task 1: Facial Classification
- **Objective**: Classify facial images into one of six categories using pre-trained convolutional neural networks (CNNs).
- **Dataset**: Facial Recognition Dataset from Kaggle (`apollo2506/facial-recognition-dataset`).
- **Models**:
  - ResNet50
  - VGG16
- **Preprocessing**:
  - Images resized to 64x64 pixels.
  - Pixel values rescaled to [0, 1].
  - Model-specific preprocessing (ResNet50 and VGG16).
- **Training**:
  - Training and validation datasets created with an 80/20 split.
  - Models fine-tuned with a custom head (GlobalAveragePooling2D, Dense layers).
  - Early stopping with patience of 5 epochs.
- **Evaluation**:
  - Test accuracy and loss computed for both models.
  - Comparison of ResNet50 and VGG16 performance.
  - Visualizations of training/validation accuracy and loss.

### Task 2: Facial Verification
- **Objective**: Verify if two facial images belong to the same person using a Siamese network.
- **Dataset**: LFW (Labeled Faces in the Wild) dataset, accessed via `sklearn.datasets.fetch_lfw_pairs`.
- **Model**:
  - Siamese network with a simple base network (Flatten, Dense layers).
  - L1 distance layer to compute feature differences.
  - Sigmoid output for binary classification (same/different person).
- **Preprocessing**:
  - Images resized to 105x105 pixels.
  - Pixel values normalized to [0, 1].
- **Training**:
  - Trained on LFW training pairs with early stopping (patience of 3 epochs).
- **Evaluation**:
  - Test accuracy and loss computed.
  - Visualizations of training/validation accuracy and loss.

## Requirements
- Python 3.x
- Libraries:
  ```bash
  numpy
  pandas
  matplotlib
  seaborn
  tensorflow
  keras
  kagglehub
  opencv-python
  scikit-learn
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
1. **Download the Dataset**:
   - The script automatically downloads the facial recognition dataset using `kagglehub`.
   - Ensure you have a Kaggle API token configured if running locally.

2. **Run the Script**:
   ```bash
   python classification_and_verification.py
   ```
   - The script will:
     - Load and preprocess the datasets.
     - Train and evaluate the classification models (Task 1).
     - Train and evaluate the Siamese network (Task 2).
     - Generate plots for model performance.

3. **Output**:
   - Classification results (test accuracy/loss for ResNet50 and VGG16).
   - Verification results (test accuracy/loss for Siamese network).
   - Plots saved as images showing accuracy and loss curves.

## File Structure
- `classification_and_verification.py`: Main Python script containing the implementation.
- `Classification_and_Verification.ipynb`: Original Jupyter Notebook with the same code.
- `README.md`: This file.

## Notes
- The script assumes access to a GPU for faster training (originally run on Colab with T4 GPU).
- Adjust `image_size`, batch size, or epochs in the script for different hardware constraints.
- The LFW dataset is automatically downloaded via `fetch_lfw_pairs`.

## License
This project is for educational purposes and uses publicly available datasets. Ensure compliance with dataset licenses (Kaggle and LFW).
