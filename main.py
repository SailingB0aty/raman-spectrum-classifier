import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_files = [
    "Data/MoSe2.csv",
    "Data/PdSe2.csv",
    "Data/PtS.csv",
    "Data/RuS.csv",
    "Data/WSe2.csv",
]

labels_to_material = [
    "MoSe2",
    "PdSe2",
    "PtS",
    "RuS",
    "WSe2",
]

X = []
Y = []

# Read CSV files and append the spectra and labels to X and Y arrays
for i, file in enumerate(data_files):
    df = pd.read_csv(file, header=None)

    # Convert the dataframe to a numpy array and transpose it so that each row represents a spectrum
    spectra = df.to_numpy().T

    # Normalize the spectra using Min-Max scaling
    spectra_min = np.min(spectra, axis=1, keepdims=True)
    spectra_max = np.max(spectra, axis=1, keepdims=True)
    spectra = (spectra - spectra_min) / (spectra_max - spectra_min)

    # Create labels for the current material
    labels = np.full(spectra.shape[0], i)

    # Append the spectra and labels to X and Y arrays
    X.append(spectra)
    Y.append(labels)

# Concatenate the arrays along the first axis
X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)

# Split the data into training and testing data at a ratio of 4:1
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Define the model architecture
model = keras.Sequential([Dense(X.shape[1], "relu", name="Input_Layer"),
                          Dense(128, "relu", name="Hidden_Layer_1"),
                          Dense(128, "relu", name="Hidden_Layer_2"),
                          Dense(len(data_files), "softmax", name="Output_Layer")
                          ])

# Train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, epochs=20, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Accuracy: {round(test_acc*100, 2)}%")

# Save the model
model.save("models/classifier.h5")

# Create schematic of model architecture
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

# Plotting the learning curves
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training and Validation Accuracy")

plt.show()


# Make predictions for the test data
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_to_material)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='viridis')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
