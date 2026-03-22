BEGIN

1. Import required libraries
   - TensorFlow / Keras
   - NumPy
   - Scikit-learn

2. Load dataset
   - Load Iris dataset
   - Separate features (X) and labels (y)

3. Split dataset
   - Divide data into training set (80%)
   - Divide data into testing set (20%)

4. Preprocess data
   - Apply standardization (mean = 0, variance = 1)

5. Build ANN model
   - Initialize Sequential model
   - Add Input layer (4 neurons)
   - Add Hidden layer 1 with ReLU activation
   - Add Hidden layer 2 with ReLU activation
   - Add Output layer with Softmax activation (3 neurons)

6. Compile model
   - Set optimizer = Adam
   - Set loss function = categorical crossentropy
   - Set evaluation metric = accuracy

7. Train model
   - Fit model using training data
   - Set epochs = 50
   - Set batch size = 16

8. Evaluate model
   - Test model using testing data
   - Calculate accuracy and loss

9. Display results
   - Print test accuracy
   - Print test loss
   - Plot accuracy graph
   - Plot loss graph

END