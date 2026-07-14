# 🔄 End-to-End ML Pipeline with Scikit-learn Pipeline API

## 🎯 Objective
The objective of this task is to design and develop a production-ready, reusable machine learning pipeline to predict customer churn using the **Telco Customer Churn Dataset**[cite: 1]. The system integrates data imputation, feature scaling, one-hot encoding, and hyperparameter tuning into a seamless scikit-learn pipeline exported via `joblib`[cite: 1].

## 🛠️ Methodology & Approach
1. **Data Preprocessing Pipeline:** Constructed a `ColumnTransformer` that handles numerical features using `SimpleImputer(median)` + `StandardScaler()` and categorical features using `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`[cite: 1].
2. **Model Training & Comparison:** Integrated preprocessing with machine learning algorithms—including `LogisticRegression` and `RandomForestClassifier`—directly within the pipeline API[cite: 1].
3. **Hyperparameter Tuning:** Applied `GridSearchCV` across 5-fold cross-validation to automatically evaluate candidate models and select the optimal hyperparameter configurations based on ROC-AUC[cite: 1].
4. **Model Export:** Persisted the tuned end-to-end pipeline using `joblib.dump()` to guarantee production reusability without data leakage[cite: 1].

## 📊 Key Results & Observations
* **Best Performing Model:** Logistic Regression (`C=0.1`, `penalty='l2'`) outperformed complex tree ensembles on this tabular dataset, demonstrating better generalization on linear feature relationships[cite: 1].
* **Test Set ROC-AUC:** Achieved **0.842** ROC-AUC and **~80.5%** overall accuracy on the test set[cite: 1].
* **Production Readiness:** Packaging data preprocessing and inference steps inside a unified `Pipeline` object guarantees identical feature transformations in production environments, preventing training-serving skew[cite: 1].
