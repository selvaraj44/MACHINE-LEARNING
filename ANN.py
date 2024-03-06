import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        self.predicted_output = self.sigmoid(self.output)
        return self.predicted_output
    
    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.hidden_error = self.output_delta.dot(self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += self.hidden_output.T.dot(self.output_delta) * self.learning_rate
        self.bias_hidden_output += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden += X.T.dot(self.hidden_delta) * self.learning_rate
        self.bias_input_hidden += np.sum(self.hidden_delta, axis=0, keepdims=True) * self.learning_rate
        
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, X):
        return self.forward(X)

# Test the Neural Network
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Define the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the neural network
    nn.train(X, y, epochs=10000)

    # Predict
    predictions = nn.predict(X)
    print("Predictions:")
    print(predictions)
