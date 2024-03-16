import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths to data
train_path = r"C:\Users\ramya\OneDrive\Desktop\main_review\static\dataset\train"
val_path = r"C:\Users\ramya\OneDrive\Desktop\main_review\static\dataset\val"
test_path = r"C:\Users\ramya\OneDrive\Desktop\main_review\static\dataset\test"

# Define preprocessing parameters
img_size = (224, 224)
batch_size = 32

# Data preprocessing
print("Preprocessing dataset...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.2, 1.0]
)

train_generator = train_datagen.flow_from_directory(train_path, 
                                                    target_size=img_size, 
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(val_path,
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=img_size,
                                                  batch_size=batch_size,
                                                  class_mode='categorical')
print()
#visualization label for images
labels = {value: key for key, value in train_generator.class_indices.items()}

print("Label Mappings for classes present in the training and validation datasets\n")
for key, value in labels.items():
    print(f"{key} : {value}")

#plotting sample training images
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
idx = 0

for i in range(2):
    for j in range(5):
        label = labels[np.argmax(train_generator[0][1][idx])]
        ax[i, j].set_title(f"{label}")
        ax[i, j].imshow(train_generator[0][0][idx][:, :, :])
        ax[i, j].axis("off")
        idx += 1

plt.tight_layout()
plt.suptitle("Sample Training Images", fontsize=21)
plt.show()
print()
# Define the CNN model architecture
print("Building CNN model...")
cnn_model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compile the CNN model
cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
cnn_model.summary()
print()
# Train the CNN model
print("Training CNN model...")
epochs = 1
history = cnn_model.fit(train_generator, 
                        epochs=epochs, 
                        validation_data=val_generator)

# Plot the accuracy and loss during training
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.show()

# Save the CNN model architecture and weights
cnn_model.save("cnn_model.h5")
print()
# Get the test accuracy and loss
test_loss, test_acc = cnn_model.evaluate(test_generator)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

# Obtain true and predicted labels for the test data
test_true_labels = test_generator.classes
test_preds = cnn_model.predict(test_generator)
test_predicted_labels = np.argmax(test_preds, axis=1)

# Print the test accuracy and classification report
print('Test accuracy:', test_acc)
print(classification_report(test_true_labels, test_predicted_labels, target_names=labels.values()))


# Generate predictions on the test data
test_preds = cnn_model.predict(test_generator)
test_classes = np.argmax(test_preds, axis=1)

# Get a random test image and its predicted class
#test_idx = np.random.randint(0, len(test_generator.filenames))
#test_img_path = os.path.join(test_path, test_generator.filenames[test_idx])
#test_img = keras.preprocessing.image.load_img(test_img_path, target_size=img_size)
#test_img_arr = keras.preprocessing.image.img_to_array(test_img)
#test_img_arr = np.expand_dims(test_img_arr, axis=0)
#predicted_class = np.argmax(cnn_model.predict(test_img_arr), axis=-1)[0]
#print("Predicted class:", predicted_class)
#print()
#print("Testing image...")
# Define paths to data
#test_path = r"C:\Users\shofi\Desktop\New folder\test,train dataset\dataset\test"

# Define preprocessing parameters
#img_size = (224, 224)

# Load the CNN model architecture and weights
#cnn_model = keras.models.load_model("cnn_model.h5")

# Load the test image
#test_img_path = r"C:\Users\shofi\Desktop\New folder\test,train dataset\dataset\test\grassland\ROIs1970_fall_s2_11_p20.png"
#test_img = keras.preprocessing.image.load_img(test_img_path, target_size=img_size)
#test_img_arr = keras.preprocessing.image.img_to_array(test_img)
#test_img_arr = np.expand_dims(test_img_arr, axis=0)

# Get the predicted class for the test image
#predicted_class = np.argmax(cnn_model.predict(test_img_arr), axis=-1)[0]

# Show the test image and its predicted class
#plt.imshow(test_img)
#plt.axis('off')
#plt.title(f"Predicted class: {predicted_class}")
#plt.show()

#getting test image path in output screen code
# Load the CNN model architecture and weights
cnn_model = keras.models.load_model("cnn_model.h5")

# Get the input path for the test image from the user
test_img_path = input("Enter the path to the test image: ")


# Load the test image
test_img = keras.preprocessing.image.load_img(test_img_path, target_size=img_size)
test_img_arr = keras.preprocessing.image.img_to_array(test_img)
test_img_arr = np.expand_dims(test_img_arr, axis=0)

# Get the predicted class for the test image
predicted_class = np.argmax(cnn_model.predict(test_img_arr), axis=-1)[0]

# Display the test image and predicted class
import matplotlib.pyplot as plt

plt.imshow(test_img)
plt.title("Predicted class: {}".format(predicted_class))
plt.axis("off")
plt.show()

print()

# Define the SNN model architecture
print("Building SNN model...")
snn_model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(224*224*3, activation='relu'),
    layers.Reshape((224, 224, 3))
])

# Convert the CNN weights to SNN weights
cnn_weights = cnn_model.get_weights()
snn_weights = []
for i in range(len(cnn_weights)):
    if i == 12: # the first dense layer
        # Initialize the SNN weights with the CNN weights and divide by 100
        # to account for the difference in scale between CNN and SNN weights
        snn_weights.append(cnn_weights[i].reshape((7, 7, 128, 1024)) / 100.)
    elif i == 14: # the second dense layer
        # Initialize the SNN weights with the CNN weights and divide by 10
        # to account for the difference in scale between CNN and SNN weights
        snn_weights.append(cnn_weights[i] / 10.)
    elif i == 16: # the last dense layer
        # Initialize the SNN weights with the CNN weights and divide by 10
        # to account for the difference in scale between CNN and SNN weights
        snn_weights.append(cnn_weights[i] / 10.)
    else: # all other layers
        snn_weights.append(cnn_weights[i])

cnn_model.set_weights(snn_weights)

# Compile the SNN model
snn_model.compile(optimizer='adam', loss='mse')

# Enhance a test image
test_img_enhanced = snn_model.predict(np.expand_dims(test_img, axis=0))[0]


# Output the results of the SNN enhancement
print("SNN Enhancement Results:")
print("Original image:")
plt.imshow(test_img)
plt.title("Original Image")
plt.show()
print("Enhanced image:")
plt.imshow(test_img_enhanced)
plt.title("Enhanced Image")
plt.show()
