import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Extract confidence scores
confidence_scores = predictions['Score'].values

# Plot histogram of confidence scores
plt.figure(figsize=(10, 6))
plt.hist(confidence_scores, bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Confidence Scores')
plt.xlabel('Confidence Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
