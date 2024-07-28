# Traffic Sign Recognition Using CNNs

This project uses Convolutional Neural Networks (CNNs) for the classification of traffic signs. The dataset consists of images of various traffic signs, and the goal is to correctly identify the type of traffic sign in each image.

## Installation

To run this project, you need to have the following libraries installed:

- `numpy` - For numerical operations
- `pandas` - For data manipulation and analysis
- `tensorflow` - For building and training deep learning models
- `os` - For interacting with the operating system (e.g., reading directories)
- `opencv-python` - OpenCV for image processing tasks
- `Pillow` - For image manipulation
- `matplotlib` - For plotting graphs and images
- `seaborn` - For plotting data distribution
- `scikit-learn` - For feature scaling, data splitting, and performance metrics

You can install these libraries using pip:

`bash`
pip install numpy pandas tensorflow opencv-python pillow matplotlib seaborn scikit-learn


## Dataset Path

Before running the code, ensure that you have set the correct path to your dataset. The dataset should be organized and placed in the specified directory. Update the `assets_path` variable in the code to point to the location of your dataset.

```python
assets_path = '/path/to/your/dataset'
```

## Data Augmentation

This project uses data augmentation to increase the diversity of the training data. We apply various augmentations using both Keras and Albumentations libraries.

### Keras Augmentations

The following augmentations are applied using Keras' `ImageDataGenerator`:

- **Rotation**: Randomly rotates the image within a specified range.
- **Width Shift**: Randomly shifts the image horizontally.
- **Height Shift**: Randomly shifts the image vertically.
- **Shear**: Randomly applies a shear transformation.
- **Zoom**: Randomly zooms into the image.
- **Horizontal Flip**: Randomly flips the image horizontally.


## Exploratory Data Analysis (EDA)

EDA is a crucial step in understanding the dataset. Here are the steps performed in the EDA:

### 1. Visualizing the Class Distribution

Understanding the distribution of classes in the dataset helps in identifying any imbalance that might exist. This is done by plotting the number of samples in each class.

### 2. Visualizing 5 Images from Each Class from the Train Set

This step involves visualizing a few samples from each class to get an idea of the variation within the classes. This helps in understanding the diversity and challenges in the dataset.

### 3. Image Dimensions

Analyzing the dimensions of the images to ensure consistency across the dataset. This helps in identifying any images that might need resizing or cropping.

### 4. Color Distribution

Examining the color distribution in the images helps in understanding the nature of the images and the importance of color information in classification.

### 5. Summary Statistics

Computing summary statistics like mean, standard deviation, etc., for the images provides insights into the dataset's characteristics and helps in preprocessing steps like normalization.

## Model Architecture

The model used for traffic sign recognition is a Convolutional Neural Network (CNN) built using Keras. Below is a description of the model architecture:

### Model Layers

1. **First Convolutional Layer**
   - 32 filters
   - Kernel size: 5x5
   - Activation: ReLU
   - Input shape: (height, width, channels) of the images

2. **Second Convolutional Layer**
   - 32 filters
   - Kernel size: 5x5
   - Activation: ReLU

3. **First Max Pooling Layer**
   - Pool size: 2x2

4. **First Dropout Layer**
   - Dropout rate: 0.25 (to prevent overfitting)

5. **Third Convolutional Layer**
   - 64 filters
   - Kernel size: 3x3
   - Activation: ReLU

6. **Fourth Convolutional Layer**
   - 64 filters
   - Kernel size: 3x3
   - Activation: ReLU

7. **Second Max Pooling Layer**
   - Pool size: 2x2

8. **Second Dropout Layer**
   - Dropout rate: 0.25 (to prevent overfitting)

9. **Flatten Layer**
   - Converts the 2D matrix to a vector

10. **Fully Connected Layer**
    - 256 units
    - Activation: ReLU

11. **Third Dropout Layer**
    - Dropout rate: 0.5 (to prevent overfitting)

12. **Output Layer**
    - 43 units (corresponding to the number of traffic sign classes)
    - Activation: Softmax

### Model Compilation

The model is compiled with the following configurations:

- **Loss function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

## Training the Model

The model is trained using the training dataset, and its performance is validated on the test dataset. Here are the steps for training the model:

### Setting the Number of Epochs

The number of epochs determines how many times the model will iterate over the entire training dataset. In this example, we train the model for 50 epochs.

### Training Process

The model is trained using the `fit` method, which takes the training data, the batch size, the number of epochs, and the validation data as inputs. The training history is stored in the `history` variable, which contains information about the training and validation loss and accuracy over each epoch.

## Model accuracy

<img width="414" alt="Screenshot 2024-07-28 at 8 40 55 AM" src="https://github.com/user-attachments/assets/4a27ace8-7be6-40c7-8699-26bd748f4ea4">

## Visualizing the Confusion Matrix using Sankey Plot

<img width="570" alt="Screenshot 2024-07-28 at 8 41 39 AM" src="https://github.com/user-attachments/assets/d9243b1b-4240-4c25-be3c-461b22a486d7">

## Conclusion

This project demonstrates the process of building a Traffic Sign Recognition system using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The steps involved include:

1. **Data Preparation**: Loading and preprocessing the dataset, including resizing images and applying data augmentation to enhance the dataset's diversity.
2. **Exploratory Data Analysis (EDA)**: Understanding the dataset through visualizations and summary statistics to gain insights and prepare for model training.
3. **Model Architecture**: Designing a CNN architecture with multiple convolutional layers, max pooling layers, dropout layers to prevent overfitting, and a final dense layer with softmax activation for classification.
4. **Model Training**: Training the model using the training dataset and validating its performance on the test dataset over multiple epochs. Monitoring the training and validation loss and accuracy to ensure the model is learning effectively.
5. **Evaluation and Analysis**: Assessing the model's performance using metrics such as accuracy, F1 score, and confusion matrix. Visualizing the model's predictions to identify areas of improvement.

By following this approach, we achieve a robust traffic sign recognition system capable of accurately classifying traffic signs into 43 different classes. This project showcases the importance of data preprocessing, careful model design, and thorough evaluation in building effective deep learning models.

### Future Work

- **Hyperparameter Tuning**: Experimenting with different hyperparameters such as learning rate, batch size, and number of epochs to further improve model performance.
- **Advanced Data Augmentation**: Implementing more sophisticated data augmentation techniques to make the model more robust to variations in the dataset.
- **Transfer Learning**: Leveraging pre-trained models to improve accuracy and reduce training time.
- **Deployment**: Deploying the trained model to a real-world application, such as an embedded system in vehicles or a web application for traffic sign recognition.

We hope this project serves as a valuable resource for understanding and implementing traffic sign recognition systems using deep learning.

## References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Albumentations Documentation](https://albumentations.ai/docs/)

## Acknowledgements

We would like to thank the providers of the dataset and the open-source community for their valuable tools and resources that made this project possible.


