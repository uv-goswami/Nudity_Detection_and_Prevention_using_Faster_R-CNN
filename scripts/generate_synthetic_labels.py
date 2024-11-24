import numpy as np
import pandas as pd

def generate_synthetic_labels(predictions_file, output_file, label_type='binary'):
    predictions = pd.read_csv(predictions_file)
    num_samples = predictions.shape[0]

    if label_type == 'binary':
        # Generate random binary labels (0 or 1)
        synthetic_labels = np.random.randint(0, 2, size=num_samples)
    elif label_type == 'multiclass':
        # Generate random multiclass labels (0, 1, 2, ...)
        num_classes = predictions['Label'].nunique()
        synthetic_labels = np.random.randint(0, num_classes, size=num_samples)
    else:
        raise ValueError("Unsupported label type. Use 'binary' or 'multiclass'.")

    # Save synthetic labels to CSV
    synthetic_labels_df = pd.DataFrame({'TrueLabel': synthetic_labels})
    synthetic_labels_df.to_csv(output_file, index=False)
    print(f"Synthetic labels saved to {output_file}")

# Example usage
generate_synthetic_labels('validation_predictions.csv', 'synthetic_labels.csv', label_type='binary')