import numpy as np
import pandas as pd

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Extract confidence scores
confidence_scores = predictions['Score'].values

# Calculate statistical summaries
mean_score = np.mean(confidence_scores)
median_score = np.median(confidence_scores)
std_score = np.std(confidence_scores)

print(f"Mean Confidence Score: {mean_score:.2f}")
print(f"Median Confidence Score: {median_score:.2f}")
print(f"Standard Deviation of Confidence Scores: {std_score:.2f}")
