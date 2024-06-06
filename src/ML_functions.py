

# Params definitions for each ML model
def get_param_dist_for_model(model_name):
    if model_name == "Random Forest":
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }
    elif model_name == "XGBoost":
        return {
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")