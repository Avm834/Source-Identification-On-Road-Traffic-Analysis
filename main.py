import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Defining batch specifications
batch_size = 32
img_height = 224
img_width = 224

## Loading training set
training_data = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/image/train',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

## Loading validation dataset
validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/image/test',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

## Loading testing dataset
testing_data = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/image/val',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

class_names = training_data.class_names
print(class_names)

## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_data = training_data.cache().prefetch(buffer_size=AUTOTUNE)
validation_data = validation_data.cache().prefetch(buffer_size=AUTOTUNE)
testing_data = testing_data.cache().prefetch(buffer_size=AUTOTUNE)

## Build the model using a pre-trained base
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Add new layers on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model
history = model.fit(training_data,
                    validation_data=validation_data,
                    epochs=20,  # You can increase the number of epochs if necessary
                    callbacks=callbacks_list)

# Serialize model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Plot training & validation accuracy and loss values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Visualize results on testing data
AccuracyVector = []
plt.figure(figsize=(20, 20))
for images, labels in testing_data.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []

    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))

    AccuracyVector = np.array(prdlbl) == labels
    for i in range(12):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: ' + predlabel[i] + ' Actl: ' + class_names[labels[i]])
        plt.axis('off')
        plt.grid(True)

# Calculate and display confusion matrix
true_labels = []
predictions_list = []

for images, labels in testing_data:
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels.extend(labels.numpy())
    predictions_list.extend(pred_labels)

# Convert lists to NumPy arrays
true_labels = np.array(true_labels)
predictions_array = np.array(predictions_list)

# Calculate and display confusion matrix
conf_mat = confusion_matrix(true_labels, predictions_array)
print("Confusion Matrix:")
print(conf_mat)

plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Display classification report
class_report = classification_report(true_labels, predictions_array, target_names=class_names)
print("Classification Report:")
print(class_report)
