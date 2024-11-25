import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Extract confidence scores
confidence_scores = predictions['Score'].values

# Define the path for the output image
output_path = 'visualizations/confidence_scores_histogram.png'

# Remove the existing image file if it exists
if os.path.exists(output_path):
    os.remove(output_path)

# Plot histogram of confidence scores
plt.figure(figsize=(10, 6))
plt.hist(confidence_scores, bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Confidence Scores')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.grid(True)

# Save the visualized image
plt.savefig(output_path)
plt.close()
