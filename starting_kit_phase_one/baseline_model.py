import pandas as pd

# Traning baseline predictions by the persistence model 
train_X_ssta = pd.read_csv('data/training_input_ssta.csv', header=None)
train_predictions = train_X_ssta
train_predictions.to_csv('out/training_output_pred_persist.csv', index=False, header=False)

# Test baseline predictions by the persistence model
test_X_ssta = pd.read_csv('data/test_input_ssta_phase_one.csv', header=None)
selected_indices = list(range(11, len(test_X_ssta), 12))
test_predictions = test_X_ssta.iloc[selected_indices]
test_predictions.to_csv('out/test_output_pred_persist_phase_one.csv', index=False, header=False)