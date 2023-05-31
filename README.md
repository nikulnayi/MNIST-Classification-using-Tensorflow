# MNIST Classification

This repository contains code for training and evaluating a deep learning model to classify handwritten digits using the MNIST dataset. The code demonstrates three different approaches to building the model: sequential, functional, and custom model classes.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The MNIST dataset is a popular benchmark dataset in the field of computer vision and machine learning. It consists of a large number of 28x28 grayscale images of handwritten digits (0-9) and corresponding labels. The goal is to train a model that can accurately classify these images into their respective digit classes.

This repository provides an implementation of the MNIST classification task using TensorFlow and Keras. It showcases three different approaches to building the model: sequential, functional, and custom model classes. Each approach demonstrates a different way of defining the model architecture and training process.

## Installation

1. Clone the repository:

```
git clone https://github.com/username/mnist_classification.git
```

2. Install the required dependencies:

```
pip install tensorflow matplotlib numpy
```

## Usage

1. Navigate to the project directory:

```
cd mnist_classification
```

2. Run the main script:

```
python main.py
```

The script will load the MNIST dataset, preprocess the data, define the model architecture, train the model, and evaluate its performance on the test set. The training and evaluation results will be displayed in the console.

## Model Architectures

This repository demonstrates three different approaches to defining the model architecture:

1. Sequential: This approach uses the `tensorflow.keras.Sequential` class to build the model. The layers are added sequentially, one after the other.

2. Functional: This approach uses the functional API of Keras to build the model. It provides more flexibility in defining complex architectures with shared layers or multiple inputs/outputs.

3. Custom Model Class: This approach defines a custom model class that inherits from `tensorflow.keras.Model`. The layers are defined as attributes of the class, and the forward pass is implemented in the `call` method.

Feel free to explore and compare these different approaches to gain a better understanding of model construction in TensorFlow and Keras.

## Dataset

The MNIST dataset is loaded using the `tensorflow.keras.datasets.mnist` module. It consists of 60,000 training images and 10,000 test images. The images are grayscale, with pixel values ranging from 0 to 255. The labels are integers representing the digit classes (0-9).

The dataset is preprocessed by normalizing the pixel values to the range of 0 to 1 and expanding the dimensions of the input arrays to match the expected shape of the models.

## Results

After training the models, the script evaluates their performance on the test set using the accuracy metric. The evaluation results, including the loss and accuracy, are displayed in the console.

The models can be further evaluated or used for predictions on new images by calling the `evaluate` or `predict` methods provided by the Keras API.

## Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes.