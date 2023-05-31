# MNIST Classification

This repository contains code for training and evaluating a deep learning model to classify handwritten digits using the MNIST dataset. The code demonstrates three different approaches to building the model: sequential, functional, and custom model classes.

## Dataset

The MNIST dataset consists of a training set of 60,000 handwritten digits images and a test set of 10,000 images. Each image is a grayscale 28x28 pixel image, representing a digit from 0 to 9.

## Getting Started

To run the code, follow these steps:

1. Clone the repository to your local machine.
2. Make sure you have the required dependencies installed. The code relies on TensorFlow, matplotlib, and numpy.
3. Run the `main.py` script using Python.

## Code Structure

The repository is organized as follows:

- `main.py`: The main script that loads the MNIST dataset, preprocesses the data, defines the model architecture, trains the model, and evaluates its performance on the test set.
- `deeplearningModels.py`: A module containing the `functional_model` and `MyCustomeModel` classes that define the model architectures using the functional and custom approaches, respectively.
- `utils.py`: A module containing utility functions, including `display_some_example` for visualizing random examples from the dataset.

## Model Architectures

The code demonstrates three different model architectures for the MNIST classification task:

1. Sequential Model: This approach uses the TensorFlow Keras `Sequential` model to define the model architecture. It consists of convolutional, pooling, batch normalization, and dense layers.

2. Functional Approach: This approach uses the TensorFlow Keras functional API to define the model architecture. It follows a similar structure to the sequential model, but the layers are defined using functional API calls.

3. Custom Model Class: This approach defines a custom model class by inheriting from the TensorFlow Keras `Model` class. The layers are defined as attributes of the class, and the model's forward pass is implemented in the `call` method.

## Usage

The `main.py` script provides an example of how to use the code to train and evaluate the model. The script performs the following steps:

1. Loads the MNIST dataset and prints the shapes of the training and test sets.
2. Preprocesses the data by normalizing the pixel values and expanding the dimensions to match the model input shape.
3. Converts the labels to one-hot encoded vectors.
4. Defines the model architecture using one of the three approaches: sequential, functional, or custom.
5. Compiles the model with an optimizer, loss function, and evaluation metrics.
6. Trains the model on the training data for a specified number of epochs.
7. Evaluates the model's performance on the test set.

Feel free to modify the code to experiment with different model architectures, hyperparameters, or training techniques.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.

## Acknowledgments

- The MNIST dataset is a widely used benchmark dataset in the field of machine learning and computer vision.
- The code in this repository is based on examples from the TensorFlow documentation and tutorials.

## Conclusion

This repository provides a straightforward implementation of a deep learning model for MNIST digit classification using different model architectures. It serves as a starting point for understanding and experimenting with image classification tasks and can be used as a reference for building more complex models.