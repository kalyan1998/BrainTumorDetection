# Brain Tumor Detection Using Pre-trained Deep Learning Models

## Project Description

This project aims to detect brain tumors from MRI images using various pre-trained deep learning models. By leveraging transfer learning, the project compares the performance of different models such as ResNet, DenseNet, InceptionV3, and others in accurately classifying brain MRI scans.

## Workflow Overview

1. **Data Collection and Preparation**
   - **Dataset**: The dataset consists of 3762 brain MRI images, sourced from Kaggle. The dataset includes 2100 images without tumors and 1662 images with tumors.
   - **Preprocessing Steps**:
     - **Cropping**: Raw images are cropped according to their boundaries to remove irrelevant parts.
     - **Resizing**: Cropped images are resized to 224x224 pixels.
     - **Normalization**: Images are normalized to have pixel values between -1 and 1.
     - **Data Augmentation**: Techniques such as flipping, rotation, and translation are used to increase the training data and avoid overfitting.

2. **Feature Extraction and Transfer Learning**
   - **Transfer Learning**: Transfer learning is utilized to leverage pre-trained models. The models used in this study include ResNet, DenseNet, InceptionV3, MobileNetV2, and VGG16.
   - **Pre-trained Models**:
     - **ResNet**: Known for its residual blocks which solve the vanishing gradient problem.
     - **DenseNet**: Uses dense connections between layers to improve feature propagation and reduce the number of parameters.
     - **InceptionV3**: Known for its efficient architecture which combines multiple convolutional filter sizes.
     - **MobileNetV2**: Optimized for mobile applications with a lightweight and efficient architecture.
     - **VGG16**: Known for its simplicity and depth, using 16 weighted layers.

3. **Model Training**
   - **Training Setup**: The models are trained using Stochastic Gradient Descent (SGD) with a learning rate optimization. The dataset is split into 80% training data (3009 images) and 20% test data (753 images).
   - **Training Process**: The models are trained for 200 epochs, and their performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.

4. **Evaluation and Results**
   - **Confusion Matrix**: A confusion matrix is used to visualize the performance of each model.
   - **Performance Metrics**: The models are compared based on their accuracy, loss, precision, recall, and F1-score.
   - **Results**: The top-performing models are ResNet50, DenseNet169, DenseNet201, InceptionV3, and MobileNetV2, with accuracies around 91%.


## Usage

### Prerequisites

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Scikit-learn

## Acknowledgments

- Jawaharlal Nehru Technological University, Hyderabad
- Dr. Supreethi K.P., Jawaharlal Nehru Technological University, Hyderabad

---

Feel free to explore the repository and use the provided tools and scripts. If you find this project useful, please give it a star!
