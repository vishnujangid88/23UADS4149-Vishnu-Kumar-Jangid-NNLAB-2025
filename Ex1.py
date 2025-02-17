import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.weights = np.zeros(input_size)  # Initialize weights as zeros
        self.bias = 0  # Initialize bias as 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for inputs, label in zip(X, y):
                # Calculate the weighted sum
                weighted_sum = np.dot(inputs, self.weights) + self.bias

                # Apply the activation function
                prediction = self.activation(weighted_sum)

                # Update weights and bias if there's an error
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

    # Predict output for new inputs
    def predict(self, X):
        predictions = []
        for inputs in X:
            weighted_sum = np.dot(inputs, self.weights) + self.bias
            predictions.append(self.activation(weighted_sum))
        return predictions

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    # Create and train the perceptron
    perceptron = Perceptron(input_size=2)
    perceptron.train(X, y)

    # Test the perceptron
    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = perceptron.predict(test_input)

    print("Predictions:")
    for i, prediction in enumerate(predictions):
        print(f"Input: {test_input[i]} => Prediction: {prediction}")
