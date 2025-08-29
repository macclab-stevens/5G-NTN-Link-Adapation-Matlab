import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Load the dataset
data_path = "Data/Case9_MCS_ThroughputCalulation_BLERw10Tbler0.01_240531_210830.csv"
logging.info(f"Loading dataset from {data_path}")
df = pd.read_csv(data_path)

# Data preprocessing: Filter for successful transmissions to learn optimal MCS
logging.info("Preprocessing data - filtering for successful transmissions")
successful_df = df[df["BLKErr"] == 1].copy()

# Round SNR to one decimal place (x.x format)
successful_df["SNR_rounded"] = successful_df["SNR"].round(1)

# For each SNR value, find the maximum MCS that still has successful transmissions
logging.info("Finding optimal MCS for each SNR value")
optimal_mcs_data = []

for snr in np.sort(successful_df["SNR_rounded"].unique()):
    snr_subset = successful_df[successful_df["SNR_rounded"] == snr]
    if not snr_subset.empty:
        # Get the highest MCS that was successfully transmitted at this SNR
        optimal_mcs = snr_subset["MCS"].max()
        
        # Get average values for other features at this SNR
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

# Create training dataset with optimal MCS values
training_df = pd.DataFrame(optimal_mcs_data)
logging.info(f"Created training dataset with {len(training_df)} SNR points")

# Features for prediction
feature_cols = [
    "SNR_rounded", "slotPercnt", "slot", "eleAnge", "PathLoss", 
    "CQI", "MOD", "TCR", "TBS", "BLER", "window", "Targetbler"
]

X = training_df[feature_cols]
y = training_df["optimal_MCS"]

# Split data into train and test sets
logging.info("Splitting data into train and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train both Linear Regression and Random Forest for comparison
logging.info("Training Linear Regression model")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

logging.info("Training Random Forest Regressor")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict with both models
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Evaluate both models
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

logging.info(f"Linear Regression - MSE: {linear_mse:.4f}, R²: {linear_r2:.4f}")
logging.info(f"Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

# Choose the better model
if linear_r2 > rf_r2:
    best_model = linear_model
    model_name = "Linear Regression"
    logging.info("Linear Regression performs better")
else:
    best_model = rf_model
    model_name = "Random Forest"
    logging.info("Random Forest performs better")

# Generate predictions for all SNR values from -8.0 to +5.0 (x.x format)
logging.info("Generating optimal MCS predictions for SNR range -8.0 to +5.0")
snr_range = np.arange(-8.0, 5.1, 0.1)
snr_range = np.round(snr_range, 1)  # Ensure x.x format

# Use median values for other features
median_features = training_df[feature_cols[1:]].median()

predictions = []
for snr in snr_range:
    # Create feature vector with current SNR and median other features
    feature_vector = [snr] + median_features.tolist()
    feature_df = pd.DataFrame([feature_vector], columns=feature_cols)
    
    predicted_mcs = best_model.predict(feature_df)[0]
    # Round to nearest valid MCS value (assuming integer MCS values)
    predicted_mcs = max(0, round(predicted_mcs))
    
    predictions.append({
        "SNR": snr,
        "Predicted_Optimal_MCS": predicted_mcs
    })
    
    if snr % 1.0 == 0:  # Log every integer SNR value
        logging.info(f"SNR: {snr:.1f}, Predicted Optimal MCS: {predicted_mcs}")

# Save results to CSV
results_df = pd.DataFrame(predictions)
results_df.to_csv("optimal_mcs_predictions.csv", index=False)
logging.info("Saved optimal MCS predictions to 'optimal_mcs_predictions.csv'")

# Feature importance (for Random Forest)
if model_name == "Random Forest":
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logging.info("Feature Importance:")
    for _, row in feature_importance.iterrows():
        logging.info(f"{row['feature']}: {row['importance']:.4f}")