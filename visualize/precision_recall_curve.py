import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from verify_clean_data import verify_clean_data

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Load synthetic labels from CSV
synthetic_labels = pd.read_csv('synthetic_labels.csv')['TrueLabel'].values
pred_scores = predictions['Score'].values

# Verify and clean data
true_labels, pred_scores = verify_clean_data(synthetic_labels, pred_scores)

# Calculate Precision-Recall Curve
precision, recall, _ = precision_recall_curve(true_labels, pred_scores)

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
