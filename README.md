# Loan Approval Prediction System

## Project Overview

This project uses machine learning to predict whether a loan application
is likely to be **Approved** or **Rejected** based on applicant and
financial details.

An interactive web application was developed using Streamlit. It allows
users to review loan application data, compare classification models,
and generate a prediction using applicant information.

## Technologies Used

-   Python
-   Pandas and NumPy
-   Matplotlib and Seaborn
-   Scikit-learn
-   XGBoost
-   Streamlit
-   Joblib

## Machine Learning Models

The project compares the following classification algorithms: -
XGBoost - Random Forest - Decision Tree - Support Vector Machine (SVM) -
Logistic Regression - K-Nearest Neighbors (KNN)

## Model Evaluation

Models are compared using: - Accuracy - Precision - Recall - F1-score

The XGBoost model achieved **98.13% test accuracy**. Its mean accuracy
across 5-fold cross-validation was **98.41%**.

## Streamlit Application Features

The application contains four sections: 1. **Overview** -- displays
total applications, approved loans, rejected loans, and loan-status
distribution. 2. **Compare Models** -- compares model performance using
accuracy, precision, recall, and F1-score. 3. **Predict Loan** --
accepts applicant details and predicts the loan status using the trained
model. 4. **Explore Data** -- displays a dataset preview, summary
statistics, and a correlation heatmap.

## Project Files

-   `app.py` -- Streamlit application.
-   `loan_approval_model.pkl` -- saved trained machine learning model.
-   `Loan_approval_dataset.csv` -- loan application dataset.
-   `Loan Approval ML.ipynb` -- Jupyter Notebook containing the machine
    learning workflow.
-   `requirements.txt` -- Python dependencies.

## How to Run the Application

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Run Streamlit

``` bash
streamlit run app.py
```

## Disclaimer

This project is developed for learning and demonstration purposes. Its
prediction should not be used as the sole basis for real-world lending
decisions.

