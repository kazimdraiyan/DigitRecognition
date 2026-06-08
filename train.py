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

nn = NeuralNetwork() # Initialized with random weights and biases


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


def differentiate_sigmoid(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2) + 1e-15


# Only the first training sample is used for demonstration
# def change_for_sample(sample_index):
#     output = evaluate_sample(sample_index)
#     expected = one_hot_y_train[sample_index]
    
#     change_w2 = np.zeros_like(nn.w2)
#     change_w1 = np.zeros_like(nn.w1)
#     change_b2 = np.zeros_like(nn.b2)
#     change_b1 = np.zeros_like(nn.b1)
    
#     learning_rate = 0.01
#     for j in range(10):
#         for k in range(64):
#             rate_C_by_z_j = differentiate_sigmoid(nn.z2[0, j]) * 2 * (output[j] - expected[j])
#             rate_C_by_w_jk = nn.a1[0, k] * rate_C_by_z_j
#             rate_C_by_b_j = rate_C_by_z_j
#             change_w2[k, j] -= learning_rate * rate_C_by_w_jk
    
#     for j in range(10):
#         rate_C_by_z_j = differentiate_sigmoid(nn.z2[0, j]) * 2 * (output[j] - expected[j])
#         change_b2[0, j] -= learning_rate * rate_C_by_z_j

#     for i in range(784):
#         for k in range(64):
#             for j in range(10):
#                 rate_C_by_z_j = differentiate_sigmoid(nn.z2[0, j]) * 2 * (output[j] - expected[j])
                
#                 rate_z_by_prev_a = nn.w2[k, j]
#                 rate_a_by_z = differentiate_sigmoid(nn.z1[0, k])
#                 rate_z_by_w_ki = x_train[sample_index, i]
#                 rate_C_by_w_ki = rate_z_by_prev_a * rate_a_by_z * rate_z_by_w_ki * rate_C_by_z_j
#                 rate_C_by_b_k = rate_z_by_prev_a * rate_a_by_z * rate_C_by_z_j
                
#                 change_w1[i, k] -= learning_rate * rate_C_by_w_ki
    
#     for j in range(10):
#         for k in range(64):
#             rate_C_by_z_j = differentiate_sigmoid(nn.z2[0, j]) * 2 * (output[j] - expected[j])
#             rate_z_by_prev_a = nn.w2[k, j]
#             rate_a_by_z = differentiate_sigmoid(nn.z1[0, k])
#             rate_z_by_b = 1
#             rate_C_by_b_k = rate_z_by_prev_a * rate_a_by_z * rate_z_by_b * rate_C_by_z_j
#             change_b1[0, k] -= learning_rate * rate_C_by_b_k
    
#     return change_w1, change_w2, change_b1, change_b2


def change_for_sample_optimized(sample_index, learning_rate=0.01):
    x = x_train[sample_index:sample_index + 1]          # (1, 784)
    y = one_hot_y_train[sample_index:sample_index + 1]  # (1, 10)

    output = evaluate_sample(sample_index).reshape(1, -1)  # (1, 10)

    # dC/dz2
    delta2 = 2 * (output - y) * differentiate_sigmoid(nn.z2)   # (1, 10)

    # dC/dw2 and dC/db2
    change_w2 = -learning_rate * (nn.a1.T @ delta2)            # (64, 10)
    change_b2 = -learning_rate * delta2                         # (1, 10)

    # dC/dz1
    delta1 = (delta2 @ nn.w2.T) * differentiate_sigmoid(nn.z1)  # (1, 64)

    # dC/dw1 and dC/db1
    change_w1 = -learning_rate * (x.T @ delta1)                 # (784, 64)
    change_b1 = -learning_rate * delta1                         # (1, 64)

    return change_w1, change_w2, change_b1, change_b2


def train_on_batch(start_index, batch_size):
    start_time = time.perf_counter()

    change_w1_sum = np.zeros_like(nn.w1)
    change_w2_sum = np.zeros_like(nn.w2)
    change_b1_sum = np.zeros_like(nn.b1)
    change_b2_sum = np.zeros_like(nn.b2)

    for i in range(start_index, start_index + batch_size):
        change_w1, change_w2, change_b1, change_b2 = change_for_sample_optimized(i, 1)
        change_w1_sum += change_w1
        change_w2_sum += change_w2
        change_b1_sum += change_b1
        change_b2_sum += change_b2

    learning_rate = 1
    nn.w1 += change_w1_sum / batch_size * learning_rate
    nn.w2 += change_w2_sum / batch_size * learning_rate
    nn.b1 += change_b1_sum / batch_size * learning_rate
    nn.b2 += change_b2_sum / batch_size * learning_rate
    
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


print("Before training:")
evaluate_sample(0, show_image=False, print_output=True)


def evaluate_loss(size):
    outputs = nn.forward(x_train[:size])
    loss = nn.mean_squared_loss(
        one_hot_y_train[:size],
        outputs
    )
    print(f"Loss: {loss:.4f}")

BATCH_SIZE = 100
for start in range(0, len(x_train), BATCH_SIZE):
    print(start // BATCH_SIZE + 1, end=": ")
    train_on_batch(start, BATCH_SIZE)
    print("Norm of w1: ", np.linalg.norm(nn.w1))
    print("Norm of w2: ", np.linalg.norm(nn.w2))
    evaluate_loss(500)

print("After training:")
evaluate_sample(0, print_output=True)
evaluate_sample(1, show_image=False, print_output=True)

evaluate_on_train_set(1000)
evaluate_on_test_set(1000)

nn.save_model("model_v02.npz")