import numpy as np
import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dropout(0.5),
  Dense(64, activation='relu'),
  Dropout(0.5),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer=SGD(lr=0.005), #optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_images,
  to_categorical(train_labels),   # one-hot vectors
  epochs=10,
  batch_size=64,
  validation_data=(test_images, to_categorical(test_labels)) #validation
)

#Testing the Model
test_loss=model.evaluate(
  test_images,
  to_categorical(test_labels)
)

print(test_loss)
#save
model.save_weights('model.h5')



#Tuning Hyperparameters
#optimizer
#the batch size and number of epochs
#Network Depth
#Activation function
#Adding Dropout layers, which are known to prevent overfitting
#Validation:use the testing dataset for validation during training.Keras will evaluate the model on the validation set at the end of each epoch and report the loss and any metrics we asked for. This allows us to monitor our model’s progress over time during training, which can be useful to identify overfitting and even support early stopping.
