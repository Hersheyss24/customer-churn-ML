import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/customer_churn_data.csv")

# Display the dataset dimensions
print("Dataset shape:", df.shape)

# Display all column names
print("\nColumn names:")
print(df.columns.tolist())

# Display data types and non-null counts
print("\nDataset information:")
df.info()

# Check missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# Count all missing values in the dataset
print("\nTotal missing values:", df.isnull().sum().sum())

# Check for duplicate rows
print("\nDuplicate rows:", df.duplicated().sum()) 

# Examine the distribution of the target variable
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Show the percentage of each class
print("\nChurn distribution (%):")
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))

# Check numerical columns for possible outliers
print("\nNumerical columns summary:")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

# Select the 12 features proposed in Milestone 1
features = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
    "Partner",
    "Dependents",
    "DeviceProtection"
]

# Separate input features (X) from the target (y)
X = df[features]
y = df["Churn"]

# Display the dimensions of X and y
print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)

# Display the first five rows of the input features
print("\nFirst five rows of input features:")
print(X.head())

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the sizes of the training and testing sets
print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

# Define numerical and categorical features
numerical_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
    "PaperlessBilling",
    "Partner",
    "Dependents",
    "DeviceProtection"
]

# Import tools for preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Create preprocessing steps for numerical features
numerical_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# Create preprocessing steps for categorical features
categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("\nPreprocessing pipeline created successfully.")

# Train a baseline Dummy Classifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DummyClassifier(strategy="most_frequent"))
])

# Train the baseline using the training data
baseline_model.fit(X_train, y_train)

# Predict churn for the unseen test data
y_pred = baseline_model.predict(X_test)

# Evaluate baseline performance
print("\n--- Baseline Model Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))


# Train a Decision Tree Classifier
decision_tree_model = Pipeline([
    ("preprocessor", preprocessor),
    DecisionTreeClassifier(random_state=42)
])

# Train the model using the training data
decision_tree_model.fit(X_train, y_train)

# Predict churn for the test data
y_pred_tree = decision_tree_model.predict(X_test)

# Evaluate Decision Tree performance
print("\n--- Decision Tree Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_tree), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred_tree, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# Train a Logistic Regression model
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model using the training data
logistic_model.fit(X_train, y_train)

# Predict churn for the test data
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate Logistic Regression performance
print("\n--- Logistic Regression Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_logistic), 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred_logistic, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred_logistic))