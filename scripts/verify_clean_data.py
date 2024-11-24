import numpy as np

def verify_clean_data(true_labels, pred_scores):

    true_labels = np.array(true_labels)
    pred_scores = np.array(pred_scores)

    if true_labels.shape[0] != pred_scores.shape[0]:
        print(f"Length mismatch: true_labels ({true_labels.shape[0]}), pred_scores ({pred_scores.shape[0]})")
        raise ValueError("The length of true_labels and pred_scores must be the same")

    if np.isnan(true_labels).any() or np.isnan(pred_scores).any():
        raise ValueError("Input arrays contain NaN values")

    if not set(np.unique(true_labels)).issubset({0, 1}):
        raise ValueError("True labels must be binary (0 and 1)")

    if not np.all((pred_scores >= 0) & (pred_scores <= 1)):
        raise ValueError("Predicted scores must be between 0 and 1")

    return true_labels, pred_scores
  
  
print("Successfully cleaned and labelled the data")
