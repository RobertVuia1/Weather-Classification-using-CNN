from google.colab import drive
import zipfile
import os


drive.mount('/content/drive')


zip_path = '/content/drive/MyDrive/ImaginiProiectCV/dataset.zip'


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')


print("Extracted Files:", os.listdir('/content/dataset'))

from google.colab import drive
drive.mount('/content/drive')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path = '/content/dataset/dataset'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')
test_dir = os.path.join(dataset_path, 'test')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Test Accuracy: {test_acc*100:.2f}%')

import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)


predicted_classes = np.argmax(predictions, axis=1)


true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(f"Total images in test dataset: {test_generator.samples}")

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


predicted_classes = []
true_classes = []


for i in range(test_generator.samples // test_generator.batch_size):
    images, labels = next(test_generator)

    predictions = model.predict(images)
    batch_predicted_classes = np.argmax(predictions, axis=1)

    predicted_classes.extend(batch_predicted_classes)
    true_classes.extend(np.argmax(labels, axis=1))

predicted_classes = np.array(predicted_classes)
true_classes = np.array(true_classes)

print("Confusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)


TP = cm[1, 1]  # True Positives (Correctly classified as positive)
TN = cm[0, 0]  # True Negatives (Correctly classified as negative)
FP = cm[0, 1]  # False Positives (Incorrectly classified as positive)
FN = cm[1, 0]  # False Negatives (Incorrectly classified as negative)

print(f"TP (True Positives): {TP}")
print(f"TN (True Negatives): {TN}")
print(f"FP (False Positives): {FP}")
print(f"FN (False Negatives): {FN}")

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

precision = precision_score(true_classes, predicted_classes, average='weighted')
recall = recall_score(true_classes, predicted_classes, average='weighted')
f1 = f1_score(true_classes, predicted_classes, average='weighted')
accuracy = accuracy_score(true_classes, predicted_classes)

print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
