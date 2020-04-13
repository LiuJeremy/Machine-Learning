import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense

test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
test_images = (test_images / 255) - 0.5

# Flatten the images.
test_images = test_images.reshape((-1, 784))


# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

# Load the model's saved weights.
model.load_weights('model.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5]) # [7, 2, 1, 0, 4]
