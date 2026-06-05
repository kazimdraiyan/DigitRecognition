import numpy as np
import time

from network.model import NeuralNetwork
import utils

print("MNIST er dataset load hocche. Meowl.")
start_time = time.perf_counter()

x_train, x_test, y_train, y_test = (
    utils.load_mnist()
)  # 60000 training samples, 10000 test samples
one_hot_y_train = utils.one_hot_encode(y_train)
one_hot_y_test = utils.one_hot_encode(y_test)

end_time = time.perf_counter()
print(f"MNIST dataset successfully loaded ({end_time - start_time:.2f} seconds)")

nn = NeuralNetwork()


# Evaluate the network on the i-th training sample
def evaluate(i, show_image=False, print_output=True):
    if show_image:
        print("Image:")
        utils.print_digit(x_train[i])
    output = nn.forward(x_train)[i]
    if print_output:
        print("Expected digit:", y_train[i])
        print("Predicted digit:", np.argmax(output))
        print("Output probabilities:", output)
    return output


# * Train only a given sample for demonstration
def change_for_sample(sample_index):
    output = evaluate(sample_index, show_image=False, print_output=False)
    
    # * New w2
    expected = one_hot_y_train[sample_index]
    change = expected - output
    current_hid = nn.a1[sample_index]  # Activation of the hidden layer

    change_w2 = np.zeros_like(nn.w2)
    for i in range(10):
        change_w2.T[i] = change[i] * current_hid  # ? Add proportionality constant?

    # print("\nAfter changing w2:")
    # evaluate(sample_index)

    # * New w1
    change_hid = np.zeros_like(current_hid)
    new_w2 = np.copy(nn.w2) + change_w2
    # Add what each neuron of the output layer thinks about the change
    for i in range(10):
        change_hid += new_w2.T[i] * change[i]

    current_input = x_train[sample_index]
    change_w1 = np.zeros_like(nn.w1)
    for i in range(64):
        change_w1.T[i] = change_hid[i] * current_input  # ? Add proportionality constant?

    # print("\nAfter changing w1:")
    # evaluate(sample_index)
    return change_w1, change_w2


def train_on_batch(start_index, batch_size=32):
    start_time = time.perf_counter()
    
    change_w1_sum = np.zeros_like(nn.w1)
    change_w2_sum = np.zeros_like(nn.w2)
    
    for i in range(start_index, start_index + batch_size):
        change_w1, change_w2 = change_for_sample(i)
        change_w1_sum += change_w1
        change_w2_sum += change_w2

    learning_rate = 0.1
    nn.w1 += change_w1_sum / batch_size * learning_rate
    nn.w2 += change_w2_sum / batch_size * learning_rate
    
    end_time = time.perf_counter()
    print(f"Batch trained in {end_time - start_time:.2f} seconds")


def evaluate_on_train_set(size=100):
    correct = 0
    for i in range(size):
        output = nn.forward(x_train)[i]
        if np.argmax(output) == y_train[i]:
            correct += 1
    accuracy = correct / size
    print(f"Training set accuracy: {accuracy:.4f}")


def evaluate_on_test_set(size=100):
    correct = 0
    for i in range(size):
        output = nn.forward(x_test)[i]
        if np.argmax(output) == y_test[i]:
            correct += 1
    accuracy = correct / size
    print(f"Test set accuracy: {accuracy:.4f}")


# print("\nTest 0 before train:")
# evaluate(0)
# change_w1, change_w2 = change_for_sample(0)
# nn.w1 += change_w1
# nn.w2 += change_w2
# print("\nTest 0 after train:")
# evaluate(0)


# print("\nTest 1 before train:")
# evaluate(1)
# change_w1, change_w2 = change_for_sample(1)
# nn.w1 += change_w1
# nn.w2 += change_w2
# print("\nTest 1 after train:")
# evaluate(1)

print("Before training:")
evaluate_on_train_set()
evaluate_on_test_set()

print("Batch 1:")
train_on_batch(0)
evaluate_on_train_set()
evaluate_on_test_set()

print("Batch 2:")
train_on_batch(32)
evaluate_on_train_set()
evaluate_on_test_set()

print("Batch 3:")
train_on_batch(64)
evaluate_on_train_set()
evaluate_on_test_set()

print("Batch 4:")
train_on_batch(96)
evaluate_on_train_set()
evaluate_on_test_set()

print("Batch 5:")
train_on_batch(128)
evaluate_on_train_set()
evaluate_on_test_set()

print("\nTest 0 after train:")
evaluate(0)
print("\nTest 1 after train:")
evaluate(1)


# for i in range(2, 50):
#     evaluate(i)
#     print()

# k = 0.1 # Proportionality constant
# change_hid = k * nn.w2.T[0] # Weights associated to the 1st neuron of the output layer
