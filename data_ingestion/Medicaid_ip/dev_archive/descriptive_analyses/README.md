# Eric Ma's Internship Repository

Welcome to my internship repo. This folder includes all of my technical contributions to the Medicaid IP model refresh. 


Here is the folder structure:
.
|
├── catboost_info 
|
├── catboost_results                              # Pipelines and results involving CatBoost implementation
│   ├── catboost_model.cbm                        # Final model object
│   ├── catboost_selected_features.txt            # List of features selected by RFE (saves time, no need run RFE again)
│   ├── catboost.ipynb                            # Model Building, Training, Optimization Pipeline
|   ├── importance_df_catboost.csv                # Feature Importances
│   ├── lift_chart_catboost.csv                   # Lift chart
|   ├── optuna_catboost.log                       # Hyperparameter optimization log
│   ├── optuna_results_catboost.csv               # Hyperparameter optimization log
|   ├── RFE_CV_Plot.png                           # RFE Plot
│   └── selected_features_and_rankings_catboost.csv # Feature Importances (unlabeled so no longer in use)
|
├── correlations                                  # Files involving correlations and some clustering
|   ├── correlation_table.csv                     # correlation values
│   ├── distinct_values.csv                       # List of features sorted by count distinct values
│   └── eric_python_start.ipynb                   # Initial data exploration and correlation generation
|
├── DNN_models
│   ├── 
|
├── prelim_descriptives
|
├── xgboost_results                               # Pipelines and results involving XGBoost implementation
│   ├── importance_df_xgboost.csv                 # Feature importances
│   ├── lift_chart_xgboost.csv                    # Lift chart
│   ├── optuna_results.csv                        # Hyperparameter optimization log
|   ├── optuna_xgboost.log                        # Hyperparameter optimization log
│   ├── xgboost_model.cbm                         # Model object
|   ├── xgboost_selected_features.txt             # List of features selected by RFE (saves time, no need run RFE again)
│   └── xgboost.ipynb                             # Model Building, Training, Optimization Pipeline
|
├── descriptives_start.ipynb                      # Descriptive study to uncover population heterogeneities
|
└── model_prelim.ipynb                            # Initial catboost testing file (no longer in use)

