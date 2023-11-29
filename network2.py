import numpy as np

# Set a random seed for reproducibility
np.random.seed(1)

# Made with assistance from ChatGPT

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Set the input data and corresponding target outputs for XOR
X = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

Y = np.array([[0],
                     [1],
                     [1],
                     [0]])

# Set the hyperparameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize weights with random values
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Training the neural network
for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Calculate the error
    error = Y - predicted_output

    # Backpropagation
    output_error = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(output_error) * learning_rate
    weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate

# Testing the trained network
test_inputs = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

hidden_layer_input = np.dot(test_inputs, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
predicted_output = sigmoid(output_layer_input)

print("Predicted output after training:")
print(predicted_output)
