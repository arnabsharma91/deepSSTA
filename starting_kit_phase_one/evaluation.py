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

def evaluate(train_y_file, baseline_file, prediction_file):
    global phase
    # Load the CSV files without headers
    train_y = pd.read_csv(train_y_file, header=None)
    baseline_predictions = pd.read_csv(baseline_file, header=None)

    # Read predictions with the same shape as train_y
    model_predictions = pd.read_csv(prediction_file, header=None)
    # Compute average RMSE for baseline predictions
    average_rmse_baseline = compute_average_rmse(baseline_predictions, train_y)
    print(f'Average RMSE (Baseline): {average_rmse_baseline}')

    # Compute average RMSE for random predictions
    average_rmse_random = compute_average_rmse(model_predictions, train_y)
    print(f'Average RMSE (Model): {average_rmse_random}')

    # Calculate the final result
    final_result = average_rmse_baseline - average_rmse_random
    print(f'Final Result (Baseline RMSE - Model RMSE): {final_result}')

    return final_result

# Example usage
if __name__ == "__main__":
    print("Phase One...")
    test_y_file = './target/data_test_output/test_output_phase_one.csv'
    baseline_file = './starting_kit_phase_one/out/test_output_pred_persist_phase_one.csv'
    prediction_file = './predictions_phase_one.csv'
    evaluate(test_y_file, baseline_file, prediction_file)
    print("\n\n")
    print("Phase Two")
    test_y_file = './target/data_test_output/test_output_phase_two.csv'
    baseline_file = './test_output_pred_persist_phase_two.csv'
    prediction_file = './predictions_phase_two.csv'
    evaluate(test_y_file, baseline_file, prediction_file)