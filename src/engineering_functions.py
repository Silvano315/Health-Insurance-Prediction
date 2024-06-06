import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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


def plot_elbow_method_with_pca(df, target, max_clusters=10):

    df_no_target = df.drop(columns=[target])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_no_target)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    inertia = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_pca)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method with PCA')
    plt.show()


def perform_kmeans_clustering(df, n_clusters, target):

    df_copy = df.copy()

    df_no_target = df_copy.drop(columns=[target])

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_no_target)

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_pca)

    df_copy['Cluster'] = clusters

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=clusters, palette='viridis', s=50, alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title('K-means Clustering with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

    return df_copy
