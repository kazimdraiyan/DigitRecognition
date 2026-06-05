import numpy as np


class NeuralNetwork:
    # One hidden layer for now. # TODO: Generalize the constructor to accept n hidden layers
    def __init__(self, input_size=28 * 28, hidden_size=64, output_size=10, seed=42):
        np.random.seed(seed)

        # Initialize with random (for now) weights and biases
        # TODO: initialize network by loading an existing model
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size) # Normal distribution. He initialization: keeps the initial weights smaller
        # TODO: Learn more
        # TODO: What are the advantages of normal distribution and He initialization?
        # TODO: Test training performance with various initialization techniques
        self.b1 = np.zeros((1, hidden_size))
        
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    # TODO: Move sigmoid and softmax to utils.py
    # Transforms R to (0, 1)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Max shifted softmax
    # Transforms the tuple z into a probability distribution: each element is between 0 and 1, all elements add up to 1
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Returns output layer, given an input layer activation
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1) # Hidden layer activation
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.softmax(self.z2) # Output layer activation
        return self.a2

    # TODO: Test with mean_squared_loss first, then compare with cross_entropy_loss
    def mean_squared_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    # TODO: Learn more
    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def print_shapes(self):
        print("W1 shape:", self.w1.shape)
        print("b1 shape:", self.b1.shape)
        print("W2 shape:", self.w2.shape)
        print("b2 shape:", self.b2.shape)
