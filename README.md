
# Transfer Learning for Binary Classification - Cats vs Dogs

Welcome to the Transfer Learning for Binary Classification GitHub repository! This project focuses on binary classification of two classes: cats and dogs, using various state-of-the-art deep learning models. The objective is to accurately classify images into either the 'cat' or 'dog' category. We have customized six base models - VGG16, VGG19, ResNet50, InceptionV3, DenseNet121, and MobileNetV2 - for this binary classification task.

Here are the updated results of our model evaluations:

|   | Model       | Test Accuracy |
|---|-------------|---------------|
| 0 | VGG16       | 98.78%        |
| 1 | VGG19       | 98.21%        |
| 2 | ResNet50    | 77.83%        |
| 3 | InceptionV3 | 99.67%        |
| 4 | DenseNet121 | 99.48%        |
| 5 | MobileNetV2 | 99.58%        |

These accuracy values represent the percentage of correctly classified images in the test dataset for each respective model.

## Overview

In this project, we explored the effectiveness of various pre-trained deep learning models on a balanced dataset containing images of cats and dogs. The models were fine-tuned and evaluated to achieve the best possible accuracy in distinguishing between the two classes.

### Image Processing Steps

1. **Array NumPy Conversion:**
   The images were converted into NumPy arrays for efficient handling in deep learning models.

2. **Reshaping to 224x224:**
   The images were reshaped to a standardized dimension of 224x224 pixels to maintain consistency across the dataset.

3. **Conversion to RGB:**
   Greyscale photos were converted to RGB format to ensure uniformity in input channels for all images.

4. **Normalization:**
   The reshaped images were normalized, scaling pixel values from 0 to 1 for optimal model training.

5. **Label Encoding:**
   Labels were encoded to represent 'cat' as 0 and 'dog' as 1 for binary classification.

6. **Train-Test Split:**
   The dataset was split into training and testing sets with an 80:20 ratio for effective model evaluation.

### Base Models

- **VGG16:**
  A widely used convolutional neural network architecture.
  
- **VGG19:**
  An extended version of VGG16 with deeper layers.
  
- **ResNet50:**
  A residual network with 50 layers, enabling the training of very deep networks.
  
- **InceptionV3:**
  A model designed to improve efficiency and accuracy through specially crafted modules.
  
- **DenseNet121:**
  A densely connected convolutional network that connects each layer to every other layer in a feed-forward fashion.
  
- **MobileNetV2:**
  A lightweight model optimized for mobile and edge devices.

### Customization and Evaluation

Each base model was customized to address the complexities of binary classification (cat or dog). Specific layers were added to adapt the models to the problem's intricacies. The customized models were then trained and evaluated using the provided dataset. The accuracy of each model was calculated to measure its performance.

## Dataset Source

The dataset used in this project can be found [here](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).

## How to Use

Feel free to explore the code to understand the customization and evaluation process of deep learning models for binary classification of cats and dogs. If you have any questions or suggestions, please don't hesitate to reach out.

Happy classifying! 🐾🐱🐶🚀
