# Health-Insurance Prediction and Explainability

1. [Introduction](#introduction)
2. [Dataset](#dataset)
   * [Kaggle](#kaggle)
   * [Description](#description)
3. [Methods and Results](#methods)
   * [EDA](#eda)
   * [Machine Learning models](#machine-learning-models)
       * [Models without oversampling](#models-without-oversampling)
       * [Models with ADASYN](#models-with-ADASYN)
   * [Explainability with ELI5](#explainability-with-ELI5)
4. [Conclusions](#conclusions)
5. [References](#references)

## Introduction

The goal of this project is not to find perfect performance, but to reach a satisfactory benchmark level in line with other projects proposed on Kaggle. Using machine learning and explainability techniques, we analyze demographic, vehicle and policy data to develop models that predict interest in vehicle insurance with high accuracy. Additionally, we add an explainability section with ELI5 to help both customers and companies in business decision making by providing insights into the factors that influence these decisions. This can help insurance companies improve their strategies to increase adoption rates and ensure more comprehensive coverage.

Vehicle insurance plays a critical role in safeguarding individuals against financial losses stemming from accidents, thefts, and other vehicle-related damages. By providing coverage for repair costs, medical expenses, and liability, vehicle insurance helps mitigate the economic impact of unforeseen events. 

According to Insurance Research Council (IRC) [[1](#ref1)][[2](#ref2)], 14 Percent of U.S. Drivers Were Uninsured in 2022.

Several factors contribute to this situation [[3](#ref3)], including:

- Lack of Awareness.
- Financial Constraints.
- Low Perceived Value.

## Dataset

### Kaggle

The dataset for this project was taken from Kaggle: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction/data

### Description

Key observations:

- Age: Skewed towards younger ages.
- Annual Premium: Positively skewed with most values concentrated at the lower end.
- Region Code: Relatively uniform distribution.
- Vintage: Nearly uniform distribution.
- Policy Sales Channel: Certain channels are used more frequently.

## Methods and Results

### EDA

Regarding the EDA methods, I analyzed the distribution of numerical features, finding a positive skew for Annual Premium and a skewed distribution for Age. I observed how categorical features varied in relation to the target feature "Response," highlighting a significant imbalance in the dataset for class 0 (as expected for cases of this nature).

After encoding the categorical features, I checked for obvious correlations between features (except for Age and Vehicle Age, which was expected).

I performed outlier detection and removal on the Annual Premium feature using Isolation Forest, which visually suggested potential outliers. Given the large dataset, I wanted to use a more sophisticated method to identify outliers. I also tried the IQR method, but it was too strict, removing many data points. I opted for a less strict approach (with a contamination rate of 0.01) because I wasn't entirely convinced that removing too many points was necessary; the input of a domain expert would have been helpful. Applying a logistic transformation would have significantly improved the point distribution. In the end, I removed only 3,798 out of over 380,000 data points.

For purely informational and visual purposes, I conducted a PCA analysis to determine how many components were needed to explain 90% of the variance, finding that six principal components were sufficient.

Finally, I applied PCA to K-means clustering to see how the data points grouped into clusters (K found using the Elbow method) using two principal components. This analysis was intended to provide insights for potential future use cases with the test dataset, which did not include the target column.

### Machine Learning models

#### Models without oversampling

The Kaggle project required evaluating the model on the [training dataset](Data/train.csv)  by calculating the area under the curve (AUC) and visualizing the ROC curve.

As an initial analysis, I split this dataset into a training set (90%) and a test set (10%). I conducted a RandomizedSearchCV for two ensemble models (robust for datasets without preprocessing and scaling) such as Random Forest and XGBoost. This technique was preferred as it didn't require excessive computational time, and performance was compared using the F1 score due to the imbalanced dataset. The performance was very good in terms of accuracy and AUC (as expected), but the models showed poor performance in metrics such as precision, recall, and F1 score, highlighting the issues of an imbalanced dataset. Specifically, performance dropped significantly on the 10% test set.

Here are some of the results:

| Set       | Model          | Accuracy | Precision | Recall  | F1 Score | AUC      |
|-----------|----------------|----------|-----------|---------|----------|----------|
| Train     | XGBoost        | 0.878245 | 0.661191  | 0.007764| 0.015348 | 0.862489 |
|           | Random Forest  | 0.999856 | 0.999735  | 0.999084| 0.999409 | 1.000000 |
| Test      | XGBoost        | 0.877231 | 0.671642  | 0.009673| 0.019072 | 0.860275 |
|           | Random Forest  | 0.865986 | 0.364252  | 0.115649| 0.175559 | 0.832948 |

#### Models with ADASYN

### Explainability with ELI5

## Conclusions

## References

1. <a name="ref1"></a> https://www.insurance-research.org/research-publications/uninsured-motorists-2
2. <a name="ref2"></a> https://www.businesswire.com/news/home/20231031282136/en/14-Percent-of-U.S.-Drivers-Were-Uninsured-in-2022-IRC-Estimates
3. <a name="ref3"></a> https://www.opportunityinstitute.org/blog/post/why-you-cant-buy-health-insurance-like-auto-insurance/






