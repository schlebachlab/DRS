This folder contains an unupervised and a series of supervised machine learning scripts designed to predict plasma membrane expression (PME) of human GPCRs using transcript and protein features.
Each suprvised machine learning scripts follows a consistent pipeline:
  1.	Load curated GPCR features dataset
  2.	Analyze PME distribution to determine an appropriate binary cutoff
  3.	Train ML models to classify GPCRs as high vs low PME. (optional) Apply hyperparameter tuning and SMOTE depending on version
  4.	Evaluate model performance 
  5.	Generate SHAP explanations and feature importances
  6.	Compare multiple baseline ML models

Supervised Machine Learning Workflow
1. Data Loading - 
Each script loads a curated dataset containing transcript and/or protein (structural and topological) features. Scripts extract only required features and create a modeling dataframe.
2. PME Distribution Analysis - 
Before creating binary labels, scripts:
  •	Compute descriptive statistics
  •	Visualize histogram + KDE
  •	Examine percentiles (10th, 25th, 33rd, median, 67th, 75th)
  •	Allow a manual cutoff provided 
  •	Report class balance and imbalance ratio
  •	Warn if the cutoff produces extreme imbalance
3. Dataset Split - 
  •	Train-test split: 80:20
  •	Stratified on the target label
  •	Seed = 42 for reproducibility
  •	Features are scaled only on the training set to prevent data leakage
4. Model Training - 
All versions train a Random Forest classifier but differs in hyperparameter optimization and resampling.
v1 — No Hyperparameter Tuning
  •	Random Forest with fixed parameters
  •	Establishes a reproducible baseline
. v2 — RandomizedSearchCV
  •	Randomized hyperparameter search for exploration of large hyperparameter space
. v3 — GridSearchCV
  •	Exhaustive hyperparameter tuning for smaller but thorough search space 
. v4 — SMOTE + GridSearchCV
  •	Oversampling using SMOTE on the training set
  •	Balances classes prior to training
  •	Followed by GridSearchCV
  •	Useful as PME cutoff creates imbalance
5. Model Evaluation - 
We report:
  •	Balanced accuracy 
  •	Precision
  •	Recall
  •	F1-score
  •	ROC-AUC
  •	Confusion matrix
  •	Full classification report
6. SHAP Interpretability - 
Each script generates:
  •	SHAP summary plot
  •	Top 10 features driving the model
7. Gini Feature Importance - 
Random Forest feature importances.
8. Baseline Model Comparison - 
Each script compares Random Forest to:
  •	Logistic Regression
  •	k-NN
  •	Decision Tree
  •	SVM (linear + RBF)
  •	Naive Bayes
  •	Gradient Boosting
  •	Neural Network (MLP)
  •	XGBoost
. Evaluation uses 5-fold cross-validation on training data with metrics:
  •	Balanced accuracy
  •	F1
  •	Precision
  •	Recall
  •	ROC-AUC

