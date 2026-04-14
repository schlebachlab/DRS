This folder contains an unupervised and a series of supervised machine learning scripts designed to predict plasma membrane expression (PME) of human GPCRs using transcript and protein features.
Each suprvised machine learning scripts follows a consistent pipeline:
  1.	Load curated GPCR features dataset
  2.	Analyze PME distribution to determine an appropriate binary cutoff
  3.	Train ML models to classify GPCRs as high vs low PME. All versions train a Random Forest classifier but differs in hyperparameter optimization and resampling. v1 — No Hyperparameter Tuning, v2 — RandomizedSearchCV, v3 — GridSearchCh, v4 — SMOTE + GridSearchCV. Final ML model used in the manuscript is from final_machine_learning_model.py that was trained on 50 random splits.
  4.	Evaluate model performance 
  5.	Generate SHAP explanations and feature importances
  6.	Compare multiple baseline ML models


 

  
