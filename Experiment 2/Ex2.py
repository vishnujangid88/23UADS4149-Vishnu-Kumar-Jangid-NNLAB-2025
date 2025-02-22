import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # Including bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        """Step activation function"""
        return 1 if x >= 0 else 0

    def predict(self, x):
        """Predict output for given input"""
        x = np.insert(x, 0, 1)  # Add bias
        return self.activation(np.dot(self.weights, x))

    def train(self, X, Y):
        """Train perceptron using Perceptron Learning Algorithm"""
        for _ in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Add bias
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = Y[i] - y_pred
                self.weights += self.learning_rate * error * x_i  # Update weights

# XOR Dataset
X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_XOR = np.array([0, 1, 1, 0])  # XOR outputs

# Train two perceptrons for the hidden layer
perceptron_nand = Perceptron(input_size=2)
perceptron_nand.train(X_XOR, np.array([1, 1, 1, 0]))  # NAND function

perceptron_or = Perceptron(input_size=2)
perceptron_or.train(X_XOR, np.array([0, 1, 1, 1]))  # OR function

# Train the output perceptron that combines NAND & OR results
hidden_outputs = np.array([[perceptron_nand.predict(x), perceptron_or.predict(x)] for x in X_XOR])
perceptron_and = Perceptron(input_size=2)
perceptron_and.train(hidden_outputs, Y_XOR)  # AND function on hidden outputs

# XOR prediction function using 3 perceptrons
def xor_using_perceptrons(x):
    nand_out = perceptron_nand.predict(x)
    or_out = perceptron_or.predict(x)
    xor_out = perceptron_and.predict([nand_out, or_out])  # Final AND output
    return xor_out

# Evaluate XOR function
correct_predictions = 0
for x, y in zip(X_XOR, Y_XOR):
    pred = xor_using_perceptrons(x)
    print(f"Input: {x}, XOR Prediction: {pred}, Actual: {y}")
    correct_predictions += int(pred == y)

accuracy = (correct_predictions / len(Y_XOR)) * 100
print(f"\nXOR Perceptron Accuracy: {accuracy:.2f}%")
