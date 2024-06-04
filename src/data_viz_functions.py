import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# Distribution analysis: Visualizing distributions of numerical features
def plot_distributions(df):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    fig.suptitle('Distribution of Numerical Features')

    age_mean = df['Age'].mean()
    age_std = df['Age'].std()

    sns.histplot(df['Age'], kde=True, ax=axes[0, 0])
    axes[0, 0].axvline(age_mean, color='r', linestyle='--', label=f'Mean: {age_mean:.2f}')
    axes[0, 0].axvline(age_mean + age_std, color='g', linestyle='--', label=f'Std: {age_std:.2f}')
    axes[0, 0].axvline(age_mean - age_std, color='g', linestyle='--')
    axes[0, 0].legend()
    axes[0, 0].set_title('Age Distribution')

    premium_mean = df['Annual_Premium'].mean()
    premium_std = df['Annual_Premium'].std()

    sns.histplot(df['Annual_Premium'], kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(premium_mean, color='r', linestyle='--', label=f'Mean: {premium_mean:.2f}')
    axes[0, 1].axvline(premium_mean + premium_std, color='g', linestyle='--', label=f'Std: {premium_std:.2f}')
    axes[0, 1].axvline(premium_mean - premium_std, color='g', linestyle='--')
    axes[0, 1].legend()
    axes[0, 1].set_title('Annual Premium Distribution')

    sns.histplot(df['Region_Code'], kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Region Code Distribution')

    sns.histplot(df['Vintage'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Vintage Distribution')

    sns.histplot(df['Policy_Sales_Channel'], kde=True, ax=axes[2, 0])
    axes[2, 0].set_title('Policy Sales Channel Distribution')

    fig.delaxes(axes[2, 1])  

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# Count and Plot the occurrences of a numerical feature with count greater than a threshold
def filter_and_plot_feature(df_train, feature_name, threshold):

  feature_counts = df_train[feature_name].value_counts()

  values_gt_threshold = feature_counts[feature_counts > threshold]

  plt.figure(figsize=(15, 8))
  sns.barplot(x=values_gt_threshold.index, y=values_gt_threshold.values, palette="viridis", hue=values_gt_threshold.index, legend=False)
  plt.title(f"{feature_name} with Counts Greater than {threshold}")
  plt.xlabel(feature_name)
  plt.ylabel('Count')
  plt.xticks(rotation=0)
  plt.show()

# Violin plots for numerical feature distirbutions
def plot_violin(df):

    numeric_cols = df.select_dtypes(exclude=['object']).columns

    numeric_cols = numeric_cols.drop(['Driving_License', 'Previously_Insured', 'Response'])

    num_cols = 3  
    num_rows = (len(numeric_cols) + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 7 * num_rows))
    axes = axes.flatten() 

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        sns.violinplot(data=df, y=col, ax=ax, color='lightblue')
        ax.set_title(f'Violin Plot for {col}')

    plt.subplots_adjust(hspace=0.3)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()


# Plot the heatmap for correlations
def plot_correlation_heatmap(corr_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()