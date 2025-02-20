import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # Including bias weight
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """Step activation function"""
        return 1 if x >= 0 else 0

    def predict(self, x):
        """Predict output for given input"""
        x = np.insert(x, 0, 1)  # Add bias input
        return self.activation(np.dot(self.weights, x))

    def train(self, X, Y):
        """Train perceptron on given dataset"""
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = Y[i] - y_pred
                self.weights += self.learning_rate * error * x_i  # Update weights

    def evaluate(self, X, Y):
        """Evaluate perceptron accuracy"""
        correct = sum(self.predict(x) == y for x, y in zip(X, Y))
        return correct / len(Y) * 100

# NAND Dataset
X_NAND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_NAND = np.array([1, 1, 1, 0])  # NAND output

# XOR Dataset
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_XOR = np.array([0, 1, 1, 0])  # XOR output

# Train perceptron for NAND
perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X_NAND, Y_NAND)
accuracy_nand = perceptron_nand.evaluate(X_NAND, Y_NAND)
print(f"NAND Perceptron Accuracy: {accuracy_nand:.2f}%")

# Train perceptron for XOR
perceptron_xor = Perceptron(input_size=2)
perceptron_xor.train(X_XOR, Y_XOR)
accuracy_xor = perceptron_xor.evaluate(X_XOR, Y_XOR)
print(f"XOR Perceptron Accuracy: {accuracy_xor:.2f}%")
