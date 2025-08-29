import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting LTE data processing and model training")

# File path
filtered_lte_path = "./Data/Case9_MCS_ThroughputCalulation_BLERw100Tbler0.1_240527_201742.csv"

# Load dataset
logging.info("Loading dataset...")
start_time = time.time()
filtered_lte_df = pd.read_csv(filtered_lte_path)
logging.info(f"Dataset loaded successfully in {time.time() - start_time:.2f} seconds")

# Filter for acceptable BLER (optional)
filtered_lte_df = filtered_lte_df[filtered_lte_df["BLER"] < 0.1]  # Only keep rows with BLER < 0.1

# Select relevant LTE features
feature_columns = ["slotPercnt", "slot", "eleAnge", "PathLoss", "SNR", "CQI", "window", "Targetbler"]
X = filtered_lte_df[feature_columns]

# Encode Grid Index labels
logging.info("Encoding Grid Index labels...")
label_encoder = LabelEncoder()
filtered_lte_df["Grid Label"] = label_encoder.fit_transform(filtered_lte_df["Grid Index"])
y = filtered_lte_df["Grid Label"]

# Train-test split
logging.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
logging.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# Train a Random Forest classifier
logging.info("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
start_time = time.time()
clf.fit(X_train, y_train)
logging.info(f"Random Forest training completed in {time.time() - start_time:.2f} seconds")

# Predictions and evaluation
logging.info("Making predictions and evaluating model...")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info("Classification Report:\n" + classification_report_str)

# Compute Confusion Matrix
logging.info("Computing Confusion Matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("./confusion_matrix.png")
plt.close()
logging.info("Confusion Matrix heatmap saved successfully")

# Compute ROC curve
logging.info("Computing ROC curve for Random Forest...")
y_score = clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)
logging.info("ROC curve computed successfully")

# Plot ROC Curve
plt.figure(figsize=(8, 4))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Random Forest")
plt.legend()
plt.grid(True)
plt.savefig("./roc_curve.png")
plt.close()
logging.info("ROC curve plot saved successfully")



# Generate heatmap for training data distribution
logging.info("Generating heatmap for training data distribution...")
grid_counts = filtered_lte_df["Grid Index"].value_counts().rename_axis("Grid Index").reset_index(name="Count")
grid_counts["Row"] = grid_counts["Grid Index"].str.extract(r'(\d+)').astype(int)
grid_counts["Column"] = grid_counts["Grid Index"].str.extract(r'([A-Za-z])')
grids = grid_counts.pivot(index="Row", columns="Column", values="Count").fillna(0)
plt.figure(figsize=(8, 6))
sns.heatmap(grids, annot=True, fmt=".0f", cmap="Reds")  # Updated fmt to handle float values
plt.xlabel("Grid Columns")
plt.ylabel("Grid Rows")
plt.title("Training Data Distribution Heatmap")
plt.savefig("./training_data_heatmap.png")
plt.close()
logging.info("Training data heatmap saved successfully")

# Feature importance
importances = clf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)
print("Importance:")
print(feature_importance_df)

logging.info("Processing and training completed successfully!")