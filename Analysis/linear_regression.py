import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Load the dataset
data_path = "Data/Case9_MCS_ThroughputCalulation_BLERw10Tbler0.01_240531_210830.csv"
logging.info(f"Loading dataset from {data_path}")
df = pd.read_csv(data_path)

# Filter for successful transmissions
logging.info("Filtering for successful transmissions (BLKErr == 1)")
successful_df = df[df["BLKErr"] == 1].copy()

# Round SNR to one decimal place
successful_df["SNR_rounded"] = successful_df["SNR"].round(1)

# For each SNR value, find the maximum MCS that was successful
logging.info("Finding optimal MCS for each SNR value (rounded to 0.1)")
optimal_mcs_data = []
for snr in np.sort(successful_df["SNR_rounded"].unique()):
    snr_subset = successful_df[successful_df["SNR_rounded"] == snr]
    if not snr_subset.empty:
        optimal_mcs = snr_subset["MCS"].max()
        avg_features = snr_subset.groupby("SNR_rounded").agg({
            "slotPercnt": "mean",
            "slot": "mean", 
            "eleAnge": "mean",
            "PathLoss": "mean",
            "CQI": "mean",
            "MOD": "mean",
            "TCR": "mean",
            "TBS": "mean",
            "BLER": "mean",
            "window": "first",
            "Targetbler": "first"
        }).iloc[0]
        optimal_mcs_data.append({
            "SNR_rounded": snr,
            "optimal_MCS": optimal_mcs,
            **avg_features.to_dict()
        })

# Create training DataFrame
training_df = pd.DataFrame(optimal_mcs_data)
logging.info(f"Created training dataset with {len(training_df)} SNR points")

# Print the training dataset
print(training_df[["SNR_rounded", "optimal_MCS", "BLER"]])

# Use only BLER as input feature
X = training_df[["BLER"]]
y = training_df["optimal_MCS"]

# Train/test split
logging.info("Splitting data into train and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
logging.info("Training Linear Regression model")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict on test set
y_pred = linear_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logging.info(f"Linear Regression - MSE: {mse:.4f}, R²: {r2:.4f}")

# Generate prediction curve for SNR values to the first decimal place
snr_range = np.arange(-8.0, 5.1, 0.1)
snr_range = np.round(snr_range, 1)

prediction_curve = []
for snr in snr_range:
    # Get BLER for this SNR from training_df, fallback to median if missing
    row = training_df[training_df["SNR_rounded"] == snr]
    if not row.empty:
        bler = row["BLER"].iloc[0]
    else:
        bler = training_df["BLER"].median()
    feature_df = pd.DataFrame([[bler]], columns=["BLER"])
    predicted_mcs = linear_model.predict(feature_df)[0]
    predicted_mcs = max(0, round(predicted_mcs))
    prediction_curve.append((snr, predicted_mcs))

# Save prediction curve to CSV
curve_df = pd.DataFrame(prediction_curve, columns=["SNR", "Predicted_Optimal_MCS"])
curve_df.to_csv("linear_regression_mcs_curve.csv", index=False)
logging.info("Saved linear regression MCS prediction curve to 'linear_regression_mcs_curve.csv'")

# Plot the prediction curve
plt.figure(figsize=(10, 6))
plt.plot(curve_df["SNR"], curve_df["Predicted_Optimal_MCS"], marker='o')
plt.xlabel("SNR (dB)")
plt.ylabel("Predicted Optimal MCS")
plt.title("Linear Regression (BLER only): Predicted Optimal MCS vs SNR")
plt.grid(True)
plt.tight_layout()
plt.savefig("linear_regression_mcs_curve.png")
logging.info("Saved plot to 'linear_regression_mcs_curve.png'")
plt.show()