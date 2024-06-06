from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Params definitions for each ML model
def get_param_dist_for_model(model_name):
    if model_name == "Random Forest":
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['log2', 'sqrt']
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
    

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, fpr, tpr, auc

def plot_roc_curves(results, set_type='train'):
    
    if set_type not in ['train', 'test']:
        raise ValueError("set_type must be either 'train' or 'test'")

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    results_dict = results
    
    if set_type == 'train':
        title = 'ROC Curve - Training Set'
    else:
        title = 'ROC Curve - Test Set'

    for name, scores in results_dict.items():
        plt.plot(scores[4], scores[5], label=f'{name} (AUC = {scores[6]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()