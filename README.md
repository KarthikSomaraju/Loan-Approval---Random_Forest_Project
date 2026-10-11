# Loan Approval Prediction using Random Forest

A machine learning project that predicts whether a loan application will be
**Approved** or **Rejected** using the Random Forest algorithm.

## Dataset
- Source: [Loan Approval Dataset (Kaggle)](https://www.kaggle.com/datasets/rohitgrewal/loan-approval-dataset)
- 4,269 rows and 13 columns
- Target: `loan_status` (Approved = 0, Rejected = 1)
- Features: dependents, education, self employment, annual income, loan amount,
  loan term, CIBIL score, and residential / commercial / luxury / bank asset values

## Workflow
1. Load the CSV and clean the column names and text values (extra spaces)
2. EDA: class balance, feature distributions, correlation heatmap, outlier check
3. Preprocessing: handle nulls, drop `loan_id`, label encoding, 80:20 train/test split
4. SMOTE on the training data only
5. Baseline Random Forest, then GridSearchCV tuning (`n_estimators`, `max_depth`, etc.)
6. Evaluation on the test set
7. Feature importance

Feature scaling was skipped because Random Forest is tree based and does not
use distances.

## Results (test set, Rejected = positive class)
| Metric | Value |
|---|---|
| Accuracy | ~0.97 |
| Precision | ~0.97 |
| Recall | ~0.95 |
| F1-score | ~0.96 |
| ROC-AUC | ~0.998 |

Best parameters: `n_estimators = 200`, `max_depth = 10`.
`cibil_score` is by far the most important feature (about 80% of the importance).

## Files
- `Random_Forest_Loan_Approval.ipynb` - the Colab notebook
- `Karthik_Somaraju_Random_Forest_Loan_Approval_Report.pdf` - the technical report
- `loan_approval_dataset.csv` - the dataset
  
## Tech stack
Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn

## Steps
1. EDA (looking at the data)
2. Preprocessing (cleaning, encoding, train test split)
3. Building the Random Forest model and choosing hyperparameters
4. Evaluating the model (confusion matrix, precision, recall, F1, ROC-AUC, feature importance)
