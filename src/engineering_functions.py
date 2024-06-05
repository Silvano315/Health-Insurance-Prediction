import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f'Number of outliers detected using IQR method: {len(outliers)}')
    
    return outliers

def outliers_isolation_forest(df, column, contamination=0.05):

    X = df[[column]].values

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_forest.fit(X)

    outlier_labels = iso_forest.predict(X)

    outliers = df[outlier_labels == -1]
    print(f'Number of outliers detected using Isolation Forest: {len(outliers)}')

    df_no_outliers = df[outlier_labels == 1]

    return df_no_outliers, outliers


def plot_pca_variance(df):

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.drop('Response', axis=1))

    pca = PCA()
    pca.fit(df_scaled)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    n_PCA_90 = np.size(cumulative_explained_variance > 0.9) - np.count_nonzero(cumulative_explained_variance > 0.9)

    # Plotting the explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(explained_variance_ratio)), explained_variance_ratio, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(len(cumulative_explained_variance)), cumulative_explained_variance, where='mid', label='Cumulative explained variance')
    plt.axvline(x=n_PCA_90, color='r', linestyle='--', label=f'{n_PCA_90} components to cover 90% of variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained Variance by Different Principal Components')
    plt.legend(loc='best')
    plt.show()
