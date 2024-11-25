import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
from verify_clean_data import verify_clean_data

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Load synthetic labels from CSV
synthetic_labels = pd.read_csv('synthetic_labels.csv')['TrueLabel'].values
pred_scores = predictions['Score'].values

# Assuming binary classification for simplicity, you might need to adjust this for multiclass
pred_labels = (pred_scores >= 0.5).astype(int)

# Verify and clean data
true_labels, pred_labels = verify_clean_data(synthetic_labels, pred_labels)

# Calculate Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

# Define the path for the output image
output_path = 'visualizations/confusion_matrix.png'

# Remove the existing image file if it exists
if os.path.exists(output_path):
    os.remove(output_path)

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title('Confusion Matrix')

# Save the visualized image
plt.savefig(output_path)
plt.close()
