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
