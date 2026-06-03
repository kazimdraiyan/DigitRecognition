import numpy as np 
from network.model import NeuralNetwork
from utils import load_mnist, one_hot_encode

# model=NeuralNetwork(

#     input_size=784,

#     hidden_size=64,

#     output_size=10,

#     seed=42

# )



# model.print_shapes()



# x=np.random.rand(1,784)



# output=model.forward(x)



# print("output shape:", output.shape)

# print("output probabilities:", output)

# print("predicted digit:", np.argmax(output))



x_train, x_test, y_train, y_test=load_mnist()



print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

y_train_one_hot = one_hot_encode(y_train)

print("y_train_one_hot shape:", y_train_one_hot.shape)
print("First label:", y_train[0])
print("First one-hot label:", y_train_one_hot[0])

nn=NeuralNetwork()

x=x_train[0].reshape(1,784)

true_label=y_train[0]

output=nn.forward(x)

predicted_label = np.argmax(output)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Output probabilities:", output)
print("True label:", true_label)
print("Predicted label:", predicted_label)

