import numpy as np
import pandas as pd
import os

# Load predictions from CSV
predictions = pd.read_csv('validation_predictions.csv')

# Extract confidence scores
confidence_scores = predictions['Score'].values

# Calculate statistical summaries
mean_score = np.mean(confidence_scores)
median_score = np.median(confidence_scores)
std_score = np.std(confidence_scores)

# Prepare summary text
summary_text = (
    f"Mean Confidence Score: {mean_score:.2f}\n"
    f"Median Confidence Score: {median_score:.2f}\n"
    f"Standard Deviation of Confidence Scores: {std_score:.2f}\n"
)

# Define the path for the output text file
output_path = 'visualizations/confidence_scores_summary.txt'

# Remove the existing text file if it exists
if os.path.exists(output_path):
    os.remove(output_path)

# Save the summary to the text file
with open(output_path, 'w') as file:
    file.write(summary_text)

print(summary_text)
