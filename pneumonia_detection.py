import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the test dataset

test_data = tf.keras.utils.image_dataset_from_directory(
    "chest_xray/test",
    image_size=(150, 150),
    batch_size=32,
    shuffle=False
)

print("Classes:", test_data.class_names)

# 2. Load the saved MobileNetV2 model

model_tl = tf.keras.models.load_model(
    "mobilenet_pneumonia.keras"
)

print("Saved MobileNetV2 model loaded!")

# 3. Test accuracy

test_loss, test_accuracy = model_tl.evaluate(test_data)

print("MobileNetV2 Test Accuracy:", test_accuracy)

# 4. Confusion Matrix and Classification Report

predictions = model_tl.predict(test_data)

predicted_classes = (predictions > 0.5).astype(int).flatten()

actual_classes = np.concatenate(
    [y for x, y in test_data],
    axis=0
)

cm = confusion_matrix(actual_classes, predicted_classes)

print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(
    classification_report(
        actual_classes,
        predicted_classes,
        target_names=["NORMAL", "PNEUMONIA"]
    )
)

# 5. Test one sample X-ray

img = tf.keras.utils.load_img(
    "sample.jpeg",
    target_size=(150, 150)
)

img_array = tf.keras.utils.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)

prediction = model_tl.predict(img_array)

if prediction[0][0] > 0.5:
    print("Sample Prediction: PNEUMONIA")
else:
    print("Sample Prediction: NORMAL")

# 6. Compare both models

models = ["Normal CNN", "MobileNetV2"]
accuracies = [71, 86]

plt.bar(models, accuracies)

plt.xlabel("Models")
plt.ylabel("Test Accuracy (%)")
plt.title("CNN vs MobileNetV2")

plt.ylim(0, 100)

plt.show()