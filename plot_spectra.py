import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = "data/graphs/PdSe2.csv"

data = []

# Read the CSV file using pandas
df = pd.read_csv(file, header=None)

# Convert the dataframe to a numpy array and transpose it so that each row represents a spectrum
spectra = df.to_numpy(dtype=np.float32).T

# Append the spectra and labels to X and Y arrays
data.append(spectra)
plt.figure(figsize=(10, 5))

labels = ["T = 550", "T = 600", "T = 650"]
for line in data[0][1:]:
    plt.plot(data[0][0], line)



plt.ylim(100, 5000)
plt.xlim(530, 550)
plt.title("PdSe2")
plt.xlabel("λ (nm)")
plt.ylabel("CCD cts")
plt.legend()
plt.show()

