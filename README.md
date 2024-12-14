# facerecognition_models
# Face Recognition with Pre-trained Models

This repository demonstrates the implementation of various deep learning models for face recognition using custom datasets. The models used include **FaceNet**, **VGGFace2 with ResNet50**, **ArcFace**, and **Squeeze-and-Excitation (SE) Networks** with **ResNet50**. The goal of this project is to fine-tune pre-trained models to classify faces into specific classes, applying data augmentation techniques to improve generalization and prevent overfitting.

## Table of Contents
1. [Objective](#objective)
2. [Libraries Used](#libraries-used)
3. [Model Architectures](#model-architectures)
    - [FaceNet](#facenet)
    - [VGGFace2 with ResNet50](#vggface2-using-resnet)
    - [ArcFace](#arcface)
    - [Squeeze-and-Excitation Networks](#se-net)
4. [Data Preprocessing](#data-preprocessing)
5. [Training](#training)
6. [Model Evaluation](#model-evaluation)
7. [Usage](#usage)
8. [Results](#results)
9. [Conclusion](#conclusion)

## Objective

The objective of this project is to fine-tune pre-trained face recognition models using custom datasets. The models aim to classify faces into 10 classes and employ data augmentation techniques for better generalization. The key models used are:

- **FaceNet**: Fine-tuned for extracting facial embeddings and classifying faces.
- **VGGFace2 with ResNet50**: Uses ResNet50 as a feature extractor, with custom layers for face classification.
- **ArcFace**: Uses ArcFace loss function for face recognition, with a custom model architecture.
- **Squeeze-and-Excitation Networks**: Incorporates SE blocks into ResNet50 to improve feature learning.

## Libraries Used

- **TensorFlow/Keras**: For building and training the neural networks.
- **keras_facenet**: For pre-trained FaceNet model.
- **ResNet50**: For pre-trained ResNet50 model used in VGGFace2 and SE Networks.
- **ImageDataGenerator**: For image preprocessing and augmentation.
- **Adam Optimizer**: Used for model optimization during training.
- **cv2**: For image loading and preprocessing.

## Model Architectures

### FaceNet
The FaceNet model is fine-tuned to generate facial embeddings and classify faces into 10 classes. The model uses a pre-trained FaceNet model, and custom classification layers are added on top. Data augmentation techniques are applied for better model generalization.

### VGGFace2 Using ResNet50
In this model, ResNet50 is used as a feature extractor, where the top layers are fine-tuned to classify faces into 10 classes. The model applies several image augmentation techniques to improve robustness and generalization.

### ArcFace
The ArcFace model optimizes face recognition by using the ArcFace loss function. This function ensures that embeddings of faces from the same person are closer, while those from different people are farther apart. The model is built using ResNet50, and a custom classification head is added.

### Squeeze-and-Excitation Networks
This model incorporates Squeeze-and-Excitation (SE) blocks with ResNet50. The SE block enhances feature learning by weighting the channels based on their importance. The model is trained on a custom dataset for face recognition.

## Data Preprocessing

The data is preprocessed using the following steps:

- **Rescaling**: Pixel values are normalized to the range [0, 1].
- **Augmentation**: Random rotation, shifting, shearing, zooming, and flipping are applied to increase the variety of the dataset and improve generalization.
- **Splitting**: The dataset is split into training and validation sets (80% training, 20% validation).

## Training

- **Epochs**: The models are trained for 20 epochs (for FaceNet and VGGFace2 models) and 10 epochs (for ArcFace and SE models).
- **Batch Size**: A batch size of 32 is used during training.
- **Optimization**: The Adam optimizer is used with a learning rate of 0.0001 for fine-tuning.
- **Loss Function**: Categorical cross-entropy is used as the loss function for multi-class classification tasks.

## Model Evaluation

The models are evaluated on the validation set, and the accuracy is reported after training. The evaluation includes:

- **Accuracy**: Measures the performance of the model on the validation dataset.
- **Confusion Matrix**: Helps visualize the performance of the classification model.

## Usage

To run the models:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/face-recognition-project.git
    cd face-recognition-project
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Prepare your dataset by organizing it into subfolders for each class inside the `data/` directory.

4. Run the training script:
    ```bash
    python train_model.py
    ```

5. After training, the model will be saved to a specified directory.

## Results

The trained models achieve high accuracy on the validation dataset, with ArcFace and VGGFace2 using ResNet50 showing strong performance in classifying faces. The data augmentation techniques significantly improved the models' generalization capabilities.

## Conclusion

This project demonstrates the power of pre-trained models in face recognition tasks. By fine-tuning models like FaceNet, ResNet50, and ArcFace, and applying advanced techniques like Squeeze-and-Excitation, we can achieve state-of-the-art results on face classification tasks. The models can be easily adapted to other datasets with similar architecture and training strategies.

---

Feel free to modify and expand this README based on the specific details of your project and repository.
