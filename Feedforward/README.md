START

1. Import required libraries

2. Load MNIST dataset
   - Split into training and testing data

3. Normalize the data
   - Divide pixel values by 255

4. Convert labels to categorical (one-hot encoding)

5. Flatten input images
   - Convert 28x28 → 784

6. Build the model
   - Add input layer
   - Add hidden layer (ReLU)
   - Add hidden layer (ReLU)
   - Add output layer (Softmax)

7. Compile the model
   - Optimizer = Adam
   - Loss = Categorical Crossentropy
   - Metrics = Accuracy

8. Train the model
   - Use training data
   - Set epochs = 10
   - Batch size = 32
   - Validate using test data

9. Evaluate the model
   - Calculate test accuracy and loss

10. Plot graphs
   - Accuracy vs Epochs
   - Loss vs Epochs

END