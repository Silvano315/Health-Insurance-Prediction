from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    
    auc = roc_auc_score(y, y_prob)
    
    fpr, tpr, _ = roc_curve(y, y_prob)
    
    return [accuracy, precision, recall, f1, auc, fpr, tpr]


def plot_metrics_comparison(comparison_df):
    # Define metrics columns
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']

    # Reset the index to get 'Train' and 'Test' as a column for easier plotting
    comparison_df_reset = comparison_df.reset_index()
    comparison_df_reset.rename(columns={'level_0': 'Set', 'level_1': 'Model'}, inplace=True)

    # Plot metrics comparison
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.barplot(x='Model', y=metric, hue='Set', data=comparison_df_reset, ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric)
        axes[i].legend(loc='best')

    # Hide the last empty subplot if the number of metrics is odd
    if len(metrics) % 2 != 0:
        axes[-1].axis('off')

    plt.tight_layout()
    plt.show()


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
        plt.plot(scores[5], scores[6], label=f'{name} (AUC = {scores[4]:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()