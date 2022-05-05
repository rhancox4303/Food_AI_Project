import os
import numpy as np
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

# Define the training and validation data folder path
directory = os.path.dirname(__file__) + "/training_validation_data"

# Define the image height and width
img_height = 256
img_width = 256

# Define the class names.
class_names = ['beans', 'cabbage', 'carrot', 'cheesecake', 'chicken wings', 'dumplings', 'french fries', 'fried rice',
               'hot dog', 'pancakes', 'pho', 'pizza', 'potato', 'radish', 'steak']

# Load the model from the Model folder.
loaded_model = tf.keras.models.load_model('Model/Food_Model')


# Method used to create the model and save it to the Model Folder
# This method only needs to be run if a new model needs to be created.
def create_model(epochs):

    # define a batch size of 32.
    batch_size = 32

    # Create training data and testing data with a 90/10 split.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Get the list of classes and the list length.
    num_classes = len(class_names)

    # adding data augmentation
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomZoom(
                0.2,
                width_factor=None,
                fill_mode='reflect',
                interpolation='bilinear',
                seed=None,
                fill_value=0.0
            ),
            layers.RandomContrast(0.5),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ]
    )

    # Define a sequential model.
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Fit the data.
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs)

    # Save the model as a folder
    model.save('Model/Food_Model')

    # Another option is to save the model as an h5 file.
    # loaded_model.save('Model/Food_Model.h5')

    # Get the validation and training accuracy.
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # The range of epochs used.
    epochs_range = range(epochs)

    # Plot the training Results.
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# Predict an image using the loaded model.
def predict_image(imagePath):
    # Load the image from the given file path.
    img = tf.keras.utils.load_img(
        imagePath, target_size=(img_height, img_width)
    )

    # Convert the image into a keras array.
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Get the prediction using the loaded model.
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get the predicted class name from the score.
    predicted_class = class_names[np.argmax(score)]

    # Create response.
    predicted_text_response = "This image is predicted to be a {}. The model predicts this with a {:.2f} percent " \
                              "confidence.". \
        format(predicted_class, 100 * np.max(score))

    # Return the response and predicted class name.
    return predicted_text_response, predicted_class


# create_model(50)
