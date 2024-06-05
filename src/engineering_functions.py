import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest

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