import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from verify_clean_data import verify_clean_data

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Load synthetic labels from CSV
synthetic_labels = pd.read_csv('synthetic_labels.csv')['TrueLabel'].values
pred_scores = predictions['Score'].values

# Verify and clean data
true_labels, pred_scores = verify_clean_data(synthetic_labels, pred_scores)

# Calculate ROC Curve
fpr, tpr, _ = roc_curve(true_labels, pred_scores)
roc_auc = roc_auc_score(true_labels, pred_scores)

# Define the path for the output image
output_path = 'visualizations/roc_curve.png'

# Remove the existing image file if it exists
if os.path.exists(output_path):
    os.remove(output_path)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {roc_auc:.2f})')

# Save the visualized image
plt.savefig(output_path)
plt.close()
