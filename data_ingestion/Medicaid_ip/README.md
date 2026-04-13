# cacm-mdcm-new_member_model
Medicaid New Member Model using Komodo data and claims embeddings is modeled on the Commercial New Member model and draws on standard code documented in the Commercial inpatient model and Medicaid claims embeddings repositories.


| #  | Product      | Dev Repository | Prod Repository  | SME | DE  |
| -- | ------------ | -------------- | ---------------- |---- |---- |
| 1  | CP IP transformer <br> (In Progress)|[cacm-cp-ip_model](https://github.aetna.com/1914536/CP_IP_transformer) | Not Started | [Jane](mailto:zouj@aetna.com)| TBD |
| 2  | Medicaid Claims Embeddings <br> (In Progress)|[cacm-mdcd-embeddings](https://github.aetna.com/analytics-org/cacm-mdcd-embeddings) | Not Started | [Elle](mailto:palmere1@aetna.com)| TBD |


Here is the folder structure:
```bash
.
├── ...
|
└── descriptive_analyses                             # This folder includes all of Eric's technical contributions to the Medicaid IP model refresh. 
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
    │   └── selected_features_and_rankings_catboost.csv # Feature Importances (unlabeled so don't use)
    |
    ├── correlations                                  # Files involving correlations and some clustering
    |   ├── correlation_table.csv                     # correlation values
    │   ├── distinct_values.csv                       # List of features sorted by count distinct values
    │   └── eric_python_start.ipynb                   # Initial data exploration and correlation generation
    |
    ├── DNN_models
    │   ├── dnn_model_v1.0_2024-08-05_16-21-03.pth    # Model object generated from model pipeline
    |   ├── NN_pipeline_v4-medicaid.ipynb             # Model Building, Training, Optimization Pipeline
    │   └── NN_pipeline.ipynb                         # Old model pipeline (don't use)
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
    └── model_prelim.ipynb                            # Initial catboost testing file (don't use)
```
