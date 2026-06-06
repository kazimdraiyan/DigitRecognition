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
def evaluate_sample(i, show_image=False, print_output=False, test=False):
    array = x_test if test else x_train
    if show_image:
        print("Image:")
        utils.print_digit(array[i])
    output = nn.forward(array[[i]])[0]  # TODO: Learn more about numpy indexing
    if print_output:
        print("Expected digit:", y_train[i])
        print("Predicted digit:", np.argmax(output))
        print("Output probabilities:", output)
    return output


# * Train only a given sample for demonstration
def change_for_sample(sample_index):
    output = evaluate_sample(sample_index)

    # * New w2
    expected = one_hot_y_train[sample_index]
    change = expected - output
    current_hid = nn.a1  # Activation of the hidden layer

    change_w2 = np.outer(current_hid, change)  # ? Add proportionality constant?

    # * New w1
    # Add what each neuron of the output layer thinks about the change
    change_hid = change @ nn.w2.T

    change_hid = change_hid.flatten()
    # change_hid /= np.linalg.norm(change_hid) # Normalize the change to prevent exploding gradients
    current_input = x_train[sample_index]
    change_w1 = np.outer(current_input, change_hid)

    return change_w1, change_w2


def evaluate_loss(size):
    outputs = nn.forward(x_train[:size])
    loss = nn.mean_squared_loss(
        one_hot_y_train[:size],
        outputs
    )
    print(f"Loss: {loss:.4f}")


def train_on_batch(start_index, batch_size):
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
    
    # # Weight decay:
    # decay_rate = 0.999
    # nn.w1 *= decay_rate
    # nn.w2 *= decay_rate

    end_time = time.perf_counter()
    print(f"Batch trained in {end_time - start_time:.4f} seconds")


def evaluate_on_train_set(size):
    correct = 0
    for i in range(size):
        output = evaluate_sample(i)
        if np.argmax(output) == y_train[i]:
            correct += 1
    accuracy = correct / size
    print(f"Training set accuracy: {accuracy * 100:.2f}%")


def evaluate_on_test_set(size):
    correct = 0
    for i in range(size):
        output = evaluate_sample(i, test=True)
        if np.argmax(output) == y_test[i]:
            correct += 1
    accuracy = correct / size
    print(f"Test set accuracy: {accuracy * 100:.2f}%")


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


# for i in range(100):
#     print("Batch", i + 1, ":")
#     train_on_batch(i)
#     evaluate_loss(500)
#     evaluate_on_train_set(500)
#     evaluate_on_test_set(500)

print("Before training:")
print("Norm of w1: ", np.linalg.norm(nn.w1))
print("Norm of w2: ", np.linalg.norm(nn.w2))
evaluate_loss(500)
evaluate_on_train_set(500)
evaluate_on_test_set(500)

print("\nTraining on batches...")

BATCH_SIZE = 100
for start in range(0, len(x_train), BATCH_SIZE):
    print(start // BATCH_SIZE + 1, end=": ")
    train_on_batch(start, BATCH_SIZE)
    print("Norm of w1: ", np.linalg.norm(nn.w1))
    print("Norm of w2: ", np.linalg.norm(nn.w2))
    evaluate_loss(500)


print("\nAfter training:")
evaluate_on_train_set(500)
evaluate_on_test_set(500)

nn.save_model("model_v01.npz")

print("\nTest 0 after train:")
evaluate_sample(0, print_output=True)
print("\nTest 1 after train:")
evaluate_sample(1, print_output=True)


# for i in range(2, 50):
#     evaluate(i)
#     print()

# k = 0.1 # Proportionality constant
# change_hid = k * nn.w2.T[0] # Weights associated to the 1st neuron of the output layer
