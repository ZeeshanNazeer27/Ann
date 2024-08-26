import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0 
test_images = test_images / 255.0
# Build the deep learning model
model = keras.Sequential([layers.Flatten(input_shape=(28, 28)),  layers.Dense(128, activation='relu'),layers.Dropout(0.2),layers.Dense(10)])
# Compile the model
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
# Train the model
ann = model.fit(train_images, train_labels, epochs=10,validation_split = 0.2)
# Evaluate the model on the test set
test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy :", test_acc)
y_prob = model.predict(test_images)
y_pred = y_prob.argmax(axis=1)
# Show the image at index 1
plt.imshow(test_images[1])
model_pred = model.predict(test_images[1].reshape(1,28,28)).argmax(axis=1)
print(model_pred)
