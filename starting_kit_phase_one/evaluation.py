import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def compute_average_rmse(predictions, labels):
    """
    Compute the average RMSE across all columns.
    """
    rmse_list = []
    for column in range(predictions.shape[1]):
        rmse = mean_squared_error(labels.iloc[:, column], predictions.iloc[:, column], squared=False)
        rmse_list.append(rmse)
    average_rmse = np.mean(rmse_list)
    return average_rmse

def evaluate(train_y_file, baseline_file):
    # Load the CSV files without headers
    train_y = pd.read_csv(train_y_file, header=None)
    baseline_predictions = pd.read_csv(baseline_file, header=None)

    # Generate random predictions with the same shape as train_y
    random_predictions = pd.DataFrame(
        np.random.uniform(-1, 1, size=train_y.shape)
    )

    # Compute average RMSE for baseline predictions
    average_rmse_baseline = compute_average_rmse(baseline_predictions, train_y)
    print(f'Average RMSE (Baseline): {average_rmse_baseline}')

    # Compute average RMSE for random predictions
    average_rmse_random = compute_average_rmse(random_predictions, train_y)
    print(f'Average RMSE (Random): {average_rmse_random}')

    # Calculate the final result
    final_result = average_rmse_baseline - average_rmse_random
    print(f'Final Result (Baseline RMSE - Random RMSE): {final_result}')

    return final_result

# Example usage
if __name__ == "__main__":
    train_y_file = 'data/training_output.csv'
    baseline_file = 'out/training_output_pred_persist.csv'

    evaluate(train_y_file, baseline_file)