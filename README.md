# customer-churn-ML

Customer Churn Prediction

1. Project Overview

For this project, I am looking at whether machine learning can be used to predict if a customer will leave a telecommunications company.

Research Question: Can customer information be used to predict whether a customer will churn?

2. Dataset

I used the Customer Churn Prediction dataset from Kaggle.
Dataset source:
https://www.kaggle.com/datasets/rashadrmammadov/customer-churn-dataset

* 5,880 rows and 21 columns.
* The target column is Churn (Yes/No).
* customerID was not used because it is only an ID.

3. Data Audit

I checked the dataset before starting the model.

* Missing values: 0
* Duplicate rows: 0
* 17 string columns, 2 integer columns and 2 float columns.

Churn distribution:

Churn	Count	Percentage
No	2,982	50.71%
Yes	2,898	49.29%

I also looked at the numerical columns: tenure, MonthlyCharges and TotalCharges. The values looked reasonable, but I did not do a separate outlier test.

4. Preprocessing

I used the 12 features selected in my project proposal.

* StandardScaler for numerical features.
* OneHotEncoder for categorical features.
* ColumnTransformer to combine both.

5. Train/Test Split

I split the data into 80% training data and 20% testing data.

* Training: 4,704 records.
* Testing: 1,176 records.

I used random_state=42 so the split can be reproduced, and stratify=y to keep the churn proportions similar in both sets.

The preprocessing was fitted on the training data to avoid data leakage.

6. 6. Model Results

I used a DummyClassifier as my baseline model. It predicts the most common class, which is No.

I also implemented a Decision Tree and Logistic Regression to compare their performance.

Model	Accuracy
DummyClassifier	50.68%
Decision Tree	48.04%
Logistic Regression	48.21%

The DummyClassifier predicted No for every customer. The other two models were able to predict both Yes and No, but their accuracy was slightly lower than the baseline.

The results show that the models still need improvement.

7. Installation and Usage

Clone the repository:

git clone https://github.com/Hersheyss24/customer-churn-ML.git
cd customer-churn-ML

Install the packages:

pip install -r requirements.txt

Run the Python file:

python3 src/milestone2.py