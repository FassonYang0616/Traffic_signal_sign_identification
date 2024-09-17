import keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Input
from Data_process_class import Process_data, Image_preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image

# Loading and Preparing Data
process_data = Process_data()
image_process = Image_preprocessing()

process_data.download_and_unzipped_file("GTSRB_Final_Training_Images.zip")
date_df = process_data.get_processed_data('image_datasets/image/GTSRB/Final_Training/Images')

crop_df = image_process.crop_image(date_df)

# This will give you a 60% training, 20% validation, and 20% test split.
train_df, test_df = train_test_split(crop_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

print(train_df.head())
print(val_df.head())

# Generate data batches for training from the training dataset.
# This generator applies data augmentation techniques such as rotation, shifts, and flips during training.
train_generator = image_process.train_data_process(train_df)

# Prepare the validation data generator.
# This generator does not apply data augmentation but still performs necessary preprocessing like rescaling.
val_generator = image_process.test_data_process(val_df)

# Prepare the test data generator similarly to the validation generator for model evaluation.
# This ensures consistency in how images are processed and presented to the model during tests.
test_generator = image_process.test_data_process(test_df)

print(train_generator)
print(val_generator)

# Use an existing ResNet50 model for transfer learning.
# Load a pre-trained ResNet50 model, excluding the top layer.
input_tensor = Input(shape=(64, 64, 3))
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Add a global average pooling layer.
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully connected layer.
x = Dense(1024, activation='relu')(x)

# Add a final softmax layer, assuming there are 43 classes.
predictions = Dense(43, activation='softmax')(x)

# Define the entire model.
model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze the weights of all convolutional layers.
# for layer in base_model.layers:
#     layer.trainable = False

# Suppose we unfreeze only the last N convolutional layers.
N = 3  # You can change this value to the number of convolutional layers you wish to unfreeze.
count = 0
for layer in base_model.layers[::-1]:  # Iterate through the model's layers from back to front.
    if isinstance(layer, Conv2D):
        layer.trainable = True
        count += 1
        if count >= N:
            break


model.compile(optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        'precision',
        'recall'
    ])

#Compile the model.
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary.
model.summary()

# Initialize the history variable.
initial_history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': []
}

for epoch in range(10):
    print(f"Epoch: {epoch + 1}/10")
    history = model.fit(
        train_generator,
        epochs=1,
        steps_per_epoch=len(train_df) // 32,
        validation_data=val_generator,
        validation_steps=len(val_df) // 32,
    )
    # Save the historical data for each epoch.
    initial_history['loss'].extend(history.history['loss'])
    initial_history['accuracy'].extend(history.history['accuracy'])
    initial_history['val_loss'].extend(history.history['val_loss'])
    initial_history['val_accuracy'].extend(history.history['val_accuracy'])

# Model Evaluation
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator, steps=len(test_df) // 32)
print('Test accuracy:', test_acc, 'Test loss:', test_loss, 'Test precision:', test_precision, 'Test recall:', test_recall)

# Plot the accuracy values for both training and validation.
plt.figure(figsize=(10, 5))
plt.plot(initial_history['accuracy'])
plt.plot(initial_history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot the loss values for both training and validation.
plt.figure(figsize=(10, 5))
plt.plot(initial_history['loss'])
plt.plot(initial_history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Ensure that matplotlib can display images in a non-interactive environment.
plt.switch_backend('agg')

# Predict on the test dataset.
test_generator.reset()  # Reset the generator to ensure the correct order of predictions.
predictions = model.predict(test_generator, steps=(len(test_df) // test_generator.batch_size) + 1)  # Ensure that all samples are predicted.
predicted_classes = np.argmax(predictions, axis=1)

# Retrieve the actual labels.
true_classes = test_generator.classes

# Identify the incorrect predictions.
errors = np.where(predicted_classes != true_classes)[0]
print(f"Number of errors = {len(errors)}/{len(true_classes)}")

# Visualize the first N incorrectly classified images.
N = 10  # Specify the number of errors to view, which can be adjusted as needed.
for i in range(min(N, len(errors))):
    error_index = errors[i]
    error_image = test_generator.filepaths[error_index]  # Retrieve the paths of the incorrectly classified images.
    error_pred_class = predicted_classes[error_index]
    error_true_class = true_classes[error_index]

    img = Image.open(error_image)
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(img)
    plt.title(f"Predicted: {error_pred_class}, True: {error_true_class}")
    plt.savefig(f'error_{i}.png')  # Save the images to a file.
    plt.close()

# Compute the confusion matrix.
conf_matrix = confusion_matrix(predicted_classes, true_classes)
print("Confusion Matrix:\n", conf_matrix)

# Generate a classification report.
target_names = [f'Class {i}' for i in range(43)]  #Generate class names based on your total number of categories.
clf_report = classification_report(predicted_classes, true_classes, target_names=target_names)
print("Classification Report:\n", clf_report)