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


# Plot the heatmap for correlations
def plot_correlation_heatmap(corr_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()