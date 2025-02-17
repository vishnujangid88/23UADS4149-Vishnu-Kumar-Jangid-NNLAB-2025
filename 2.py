import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward(X, W1, b1, W2, b2):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    return hidden_output, final_output

def backward(X, y, W1, b1, W2, b2, hidden_output, final_output, lr):
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)
    
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
    
    return W1, b1, W2, b2

def train_mlp(X, y, hidden_neurons=2, epochs=10000, lr=0.5):
    np.random.seed(42)
    input_neurons, output_neurons = X.shape[1], 1
    W1 = np.random.uniform(-1, 1, (input_neurons, hidden_neurons))
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.uniform(-1, 1, (hidden_neurons, output_neurons))
    b2 = np.zeros((1, output_neurons))
    
    for _ in range(epochs):
        hidden_output, final_output = forward(X, W1, b1, W2, b2)
        W1, b1, W2, b2 = backward(X, y, W1, b1, W2, b2, hidden_output, final_output, lr)
    
    return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, final_output = forward(X, W1, b1, W2, b2)
    return np.round(final_output)

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train MLP
W1, b1, W2, b2 = train_mlp(X, y)

# Test MLP
predictions = predict(X, W1, b1, W2, b2)
print("Predictions:", predictions.flatten())
