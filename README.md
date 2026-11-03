# customer-churn-ML

Customer Churn Prediction

1. Project Overview

This project investigates whether customer information can be used to predict customer churn in a telecommunications company using machine learning.

Research Question: Can customer information be used to predict whether a telecommunications customer will churn?

2. Dataset

The dataset, Customer Churn Prediction, was obtained from Kaggle.

* 5,880 records and 21 columns.
* Target variable: Churn (Yes/No).
* customerID is excluded as it is an identifier.

3. Data Audit

The initial data audit found:

* Missing values: 0
* Duplicate rows: 0
* Variable types: 17 string, 2 integer and 2 float columns.

Churn	Count	Percentage
No	2,982	50.71%
Yes	2,898	49.29%

Numerical summaries were examined for tenure, MonthlyCharges and TotalCharges.

I also checked the ranges of the numerical columns. The values looked reasonable, although I haven’t carried out a separate outlier test.

4. Preprocessing

Twelve features from the original project proposal were selected.

* StandardScaler was applied to numerical features.
* OneHotEncoder was applied to categorical features.
* ColumnTransformer combined both preprocessing steps.

5. Train/Test Split

The dataset was split into:

* 80% training data (4,704 records).
* 20% testing data (1,176 records).

Stratification was used to maintain class proportions, with random_state=42 for reproducibility. The preprocessing pipeline was fitted on training data only to prevent data leakage.

6. Baseline Model

A DummyClassifier using the most_frequent strategy was used as the baseline.

Metric	Result
Accuracy	50.68%
Recall (No)	1.00
Recall (Yes)	0.00

The model predicted every customer as No. It therefore failed to identify customers who churned, providing a baseline for comparison with future models.

7. Installation and Usage

Clone the repository:

git clone https://github.com/Hersheyss24/customer-churn-ML.git
cd customer-churn-ML

Install dependencies:

pip install -r requirements.txt

Run the project:

python3 src/milestone2.py
