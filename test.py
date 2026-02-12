import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

model = keras.models.load_model("models/classifier.h5")
file = "Test data/WSe2-700.csv"

labels_to_material = [
    "MoSe2",
    "PdSe2",
    "PtS",
    "RuS",
    "WSe2",
]
true_label = "WSe2"

data = []

# Read the CSV file using pandas
df = pd.read_csv(file, header=None)

# Convert the dataframe to a numpy array and transpose it so that each row represents a spectrum
spectra = df.to_numpy().T

# Normalize the spectra using Min-Max scaling
spectra_min = np.min(spectra, axis=1, keepdims=True)
spectra_max = np.max(spectra, axis=1, keepdims=True)
spectra = (spectra - spectra_min) / (spectra_max - spectra_min)

# Append the spectra and labels to X and Y arrays
data.append(spectra)



# Use the trained model to make predictions on the test set
y_pred = model.predict(spectra)
y_pred_classes = np.argmax(y_pred, axis=1)

# Display each spectrum in the test set along with its predicted and actual values
for i, spectrum in enumerate(spectra):
    predicted_value = y_pred_classes[i]
    print(f"Predicted: {labels_to_material[predicted_value]}    Actual: {true_label}")