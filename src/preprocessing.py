import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

RAW_DIR = "data/raw"
FILE_NAME = "CICIoT2023.csv"
PROCESSED_DIR = "data/processed"

# Load the dataset
data = pd.read_csv(RAW_DIR + "/" + FILE_NAME)

# Clean the dataset
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Separate features and labels
X = data.drop(columns=['Label'])
y = data['Label']

# Binary encoding of labels: 0 for 'BENIGN', 1 for all other labels
y_binary = y.apply(lambda x: 0 if x == 'BENIGN' else 1)

# Feature scaling
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Normal and anomalous data split
X_normal = X_scaled[y_binary == 0]
X_anomaly = X_scaled[y_binary == 1]

# Train-test split
X_train, X_test_normal = train_test_split(X_normal, test_size=0.2, random_state=42)
X_test = np.concatenate((X_test_normal, X_anomaly), axis=0)
y_test = np.concatenate((np.zeros(len(X_test_normal)), np.ones(len(X_anomaly))), axis=0)

# Save the processed data - iForest, SVM, Autoencoder
np.save(PROCESSED_DIR + "/baseline/X_train.npy", X_train)
np.save(PROCESSED_DIR + "/baseline/X_test.npy", X_test)
np.save(PROCESSED_DIR + "/baseline/y_test.npy", y_test)

# For OpenMax
le = LabelEncoder()
y_encoded = le.fit_transform(y)
np.save(PROCESSED_DIR + "/openmax/X.npy", X_scaled)
np.save(PROCESSED_DIR + "/openmax/y.npy", y_encoded)
np.save(PROCESSED_DIR + "/openmax/classes.npy", le.classes_)

print("Data preprocessing complete. Processed files saved in:", PROCESSED_DIR)