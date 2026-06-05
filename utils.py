import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# TODO: Organize them as static methods into classes


def load_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    x = mnist.data
    y = mnist.target.astype(int)
    x = x / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=10000, random_state=42, stratify=y
    ) # stratify ensures that the class distribution is preserved in both training and test sets

    return x_train, x_test, y_train, y_test


# For example: converts a digit label 3 to a one-hot encoded vector [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot


# Transforms R to (0, 1)
def sigmoid(z):
        return 1 / (1 + np.exp(-z))


# Max shifted softmax
# Transforms the tuple z into a probability distribution: each element is between 0 and 1, all elements add up to 1
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# Print an input activation as a human readable digit
def print_digit(x, threshold=0.5):
    for i in range(28):
        line = "".join("*" if x[i * 28 + j] > threshold else " " for j in range(28))
        if line.strip():
            print(line)
