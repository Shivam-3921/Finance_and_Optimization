# Credit Default Prediction: Benchmarking Optimization Algorithms

This project trains a neural network to predict consumer credit default risk, using the task as a testbed to benchmark how different gradient-based optimization algorithms affect training convergence and final model performance.

## Dataset

["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) — a real-world credit risk dataset of ~150,000 borrower records with 10 financial and demographic features (revolving credit utilization, age, past delinquencies, debt ratio, income, number of open credit lines, real estate loans, dependents), with the target being whether the borrower experienced serious delinquency within two years.

## Approach

- **EDA & preprocessing:** distribution and outlier analysis (histograms, box plots) on all features; removed unrealistic outliers (e.g. implausible utilization ratios, debt ratios, delinquency counts); standardized features with `StandardScaler`
- **Model:** a feedforward neural network (`Linear → ReLU → BatchNorm` stack) built in PyTorch for binary classification
- **Optimizer benchmarking:** trained the identical architecture under six optimization configurations to isolate the effect of the optimizer itself on convergence speed and accuracy:
  - SGD (no momentum)
  - SGD with momentum (β = 0.9, 0.7, 0.5, 0.3, 0.1)
  - AdaGrad
  - RMSProp
  - Adam
- **Evaluation:** test accuracy, confusion matrix, and full classification report (precision/recall/F1) for each optimizer, plus training loss curves compared across optimizers

## Results

All optimizers converged to a similar test accuracy in the **93.2–93.7% range**, with Adam-family adaptive methods (RMSProp, Adam) reaching low training loss in noticeably fewer epochs than vanilla SGD — illustrating the practical trade-off between per-step cost and convergence speed across optimizer choices on the same architecture and data.

## Tech Stack

Python, PyTorch, scikit-learn, Pandas, Seaborn, Matplotlib

## Files

- `code.ipynb` — full pipeline: EDA, preprocessing, model definition, and optimizer benchmarking
- `cleaned_dataset.csv` — preprocessed/standardized dataset used for training
- `data/` — original Kaggle competition files (training set, test set, data dictionary)
- `results/` — saved output plots

## Note

`winter.ipynb` in this repo is a separate exploratory notebook not part of the core analysis described above.
