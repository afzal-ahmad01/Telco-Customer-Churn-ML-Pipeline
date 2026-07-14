# ==========================================
# 1. SETUP & DATASET LOADING[cite: 1]
# ==========================================
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Load Telco Customer Churn dataset[cite: 1]
# Assumes CSV is downloaded: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
print("Loading Telco Customer Churn Dataset...")
df = pd.read_csv(url)

# Clean Target Variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop(columns=['customerID'], inplace=True, errors='ignore')

# Handle whitespace values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

X = df.drop('Churn', axis=1)
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 2. PREPROCESSING PIPELINE CONSTRUCTION[cite: 1]
# ==========================================
# Automatically identify numerical and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric Features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

# Create numeric preprocessing pipeline: Impute missing values -> Scale[cite: 1]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create categorical preprocessing pipeline: Impute missing values -> OneHotEncode[cite: 1]
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers into a single ColumnTransformer[cite: 1]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 3. MODEL TRAINING & HYPERPARAMETER TUNING[cite: 1]
# ==========================================
# Construct complete pipeline with a placeholder classifier
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid covering Logistic Regression and Random Forest[cite: 1]
param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 15, None],
        'classifier__min_samples_split': [2, 5]
    }
]

print("Executing GridSearchCV for Hyperparameter Tuning...")
grid_search = GridSearchCV(
    full_pipeline, 
    param_grid, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train, y_train)

# ==========================================
# 4. EVALUATION & INSIGHTS[cite: 1]
# ==========================================
best_model = grid_search.best_estimator_
print(f"\nBest Model Selected via GridSearch: {grid_search.best_params_['classifier']}[cite: 1]")
print(f"Best Cross-Validation ROC-AUC Score: {grid_search.best_score_:.4f}")

# Evaluate on unseen test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Test Set Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}[cite: 1]")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 5. MODEL EXPORT & REUSABILITY[cite: 1]
# ==========================================
export_filename = 'customer_churn_pipeline.pkl'
joblib.dump(best_model, export_filename)
print(f"\nSuccessfully exported production pipeline to: '{export_filename}'[cite: 1]")

# Verify reusability by reloading and predicting on sample data[cite: 1]
loaded_pipeline = joblib.load(export_filename)
sample_prediction = loaded_pipeline.predict(X_test.iloc[:5])
print(f"Verification Sample Predictions from reloaded pipeline: {sample_prediction}")