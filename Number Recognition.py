import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the neural network architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', test_accuracy)

# Make predictions on new handwritten images
def predict_digit(image):
  """Predicts the digit in a handwritten image.

  Args:
    image: A 28x28 grayscale image of a handwritten digit.

  Returns:
    The predicted digit.
  """

  image = image.reshape((1, 28, 28))
  predictions = model.predict(image)
  return np.argmax(predictions[0])

# Example usage:

image = X_test[0]
prediction = predict_digit(image)

print('Predicted digit:', prediction)
