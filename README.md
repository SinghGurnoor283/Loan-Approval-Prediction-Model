# Loan-Approval-Prediction-Model
This project contains a machine learning model and Streamlit web application to predict whether a loan application will be Approved or Rejected  based on user inputs like income, credit score, asset value, and more.

##  Features

- Built using **multiple machine learning algorithms**:
  - ✔️ Logistic Regression
  - ✔️ Support Vector Machine (SVM)
  - ✔️ Random Forest (with GridSearchCV hyperparameter tuning)
- Automatically selects the best-performing model (**Random Forest** in this case).
- Scales features using **StandardScaler**.
- Real-time prediction via an interactive **Streamlit** web app.
- Model performance (accuracy) is displayed.

##  Model Performance

| Model                | Accuracy (Test Set) |
|----------------------|---------------------|
| Logistic Regression  | ~90.2%              |
| SVM (Linear Kernel)  | ~91.0%              |
| Random Forest (Best) | ~98.8%              |

👉 **Random Forest** was selected for deployment due to its superior accuracy.


##  Files in This Repository

| File                | Description                            |
|---------------------|----------------------------------------|
| `loan_model.pkl`    | Trained Random Forest ML model         |
| `scaler.pkl`        | StandardScaler used in training        |
| `app.py`            | Streamlit web application              |
| `model_columns.pkl` | Column order used during training      |
| `README.md`         | Project overview                       |


##  Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
