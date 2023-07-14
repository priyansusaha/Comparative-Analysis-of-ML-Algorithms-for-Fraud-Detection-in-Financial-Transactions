# Comparative-Analysis-of-ML-Algorithms-for-Fraud-Detection-in-Financial-Transactions
Research Paper - https://drive.google.com/file/d/1kcVYgelm6ik-sbGAptJAbzO2_5NvdXtf/view
## Introduction
Fraud detection in financial transactions has become a critical area of concern due to the escalating complexity and sophistication of fraudulent activities. In this project, I delve into the exploration of machine learning techniques to combat fraud in financial transactions. By leveraging a synthetic dataset encompassing 6,362,620 transactions, I analyze the performance of various machine learning classifiers using transaction-level features. Specifically, I used Decision Trees, K-Nearest Neighbors, Logistic Regression, Gaussian Naive Bayes, and Random Forests classifiers to identify the most effective approach. I assess the performance of each classifier and compare their results, focusing on accuracy as the primary evaluation metric. The findings indicate that Random Forest Classifier surmounts the other classifiers with an accuracy rate of 99% and recall of 100%, by leveraging a combination of ensemble learning and decision trees, the Random Forest Classifier showcased superior predictive capabilities and robustness, outperforming the other classifiers examined in this study. Its ability to effectively capture complex relationships and identify subtle patterns within the transaction-level features contributed to its outstanding performance, demonstrating its superior predictive capabilities for fraud detection in financial transactions.
## Dataset Description
The dataset utilized in this project was sourced from the "Synthetic Financial Datasets for Fraud Detection" repository available on Kaggle (https://www.kaggle.com/datasets/ealaxi/paysim1). This repository specifically focuses on providing synthetic financial data for fraud detection research. It should be noted that the dataset used in this study represents a subset of the original dataset, consisting of approximately 6,362,620 samples. 
Within this dataset, there were a total of 8213 fraud transactions and 6,354,407 legitimate ones. This distribution enabled the study to capture a wide range of transactional scenarios, including both fraudulent and legitimate activities. By analyzing this dataset, I aimed to gain insights into the performance of various machine learning classifiers in effectively detecting instances of fraud within financial transactions.
## Analysis of the data
As part of my project, I conducted a comprehensive examination of the dataset to uncover any underlying patterns and potential correlations among its features. The main objective of this analysis was to identify noteworthy features or information that could aid in effectively distinguishing fraudulent and non-fraudulent transactions. To begin the analysis, I loaded the complete dataset and utilized the Matplotlib library to create a heatmap showcasing the correlation matrix. This visual representation allowed for a clearer understanding of the interrelationships between different features.

<p align="center">
  <img width="460" height="300" src="https://github.com/priyansusaha/Comparative-Analysis-of-ML-Algorithms-for-Fraud-Detection-in-Financial-Transactions/assets/26963104/8d60e0fd-7eae-4d81-a1a3-fceb3838523b">
</p>

In this analysis, I focused on selecting features with correlation values greater than 0.4 to generate the heatmap. After plotting the correlation matrix in the form of a heatmap, I observed that certain features displayed significant correlations. These included oldbalanceOrg, newbalanceOrg, oldbalanceDest, and newbalanceDest. It is worth noting that these correlations were apart from the diagonal entries, which represent the correlation of a feature with itself. To gain a better understanding of the feature distributions, I proceeded to create a scatter density plot for further analysis.
## Evaluation of the models
I used the scikit-learn library for evaluating the performance of the machine-learning models. Scikit-learn is a powerful and widely used Python library that provides a comprehensive set of tools for machine learning, including classification and evaluation metrics. The evaluation results for each model are summarized below:
### I. K-Nearest Neighbors (KNN)
The KNN model achieved an accuracy of 96% on the dataset. It exhibited high precision (95%) and recall (97%) for classifying both fraudulent and non-fraudulent transactions. The F1-score, which represents the balance between precision and recall, was also high at 96%. These results indicate that the KNN model performed well in accurately identifying fraudulent transactions.
### II. Logistic Regression
The logistic regression model achieved an accuracy of 82% on the dataset. It displayed a precision of 74% and recall of 98% for detecting fraudulent transactions. However, the precision and recall for non-fraudulent transactions were 97% and 66%, respectively. The F1-score was relatively lower at 85%. These findings suggest that the logistic regression model had a higher tendency to classify non-fraudulent transactions correctly, while it exhibited a lower precision for fraudulent transactions.
### III. Naive Bayes
The naive Bayes model achieved an accuracy of 63% on the dataset. It showed a precision of 88% for detecting fraudulent transactions but had a lower recall of 30%. The precision for non-fraudulent transactions was 57%, and the recall was 96%. The F1-score for the model was 45%. These results indicate that the naive Bayes model had a relatively higher precision for fraudulent transactions but struggled with recall, resulting in a lower overall accuracy.
### IV. Decision Tree
The decision tree model achieved a high accuracy of 99% on the dataset. It displayed excellent precision and recall scores of 99% for both fraudulent and non-fraudulent transactions. The F1-score was also high at 99%. These results indicate that the decision tree model performed exceptionally well in accurately classifying transactions as fraudulent or non-fraudulent.
### V. Random Forest
The random forest model achieved a high accuracy of 99% on the dataset. It demonstrated a precision of 99% and recall of 100% for detecting both fraudulent and non-fraudulent transactions. The F1-score was 99%. These findings suggest that the random forest model performed remarkably well in accurately classifying transactions.

<p align="center">
  <img width="460" height="300" src="https://github.com/priyansusaha/Comparative-Analysis-of-ML-Algorithms-for-Fraud-Detection-in-Financial-Transactions/assets/26963104/499f6cc4-95fd-4876-8692-84c62fe3baa1">
</p>

## ROC Curve Analysis

<p align="center">
  <img width="460" height="300" src="https://github.com/priyansusaha/Comparative-Analysis-of-ML-Algorithms-for-Fraud-Detection-in-Financial-Transactions/assets/26963104/713e2542-4f42-4f11-9273-aa4500f9f7fd">
</p>

Based on the ROC curve analysis, the decision tree and random forest models exhibited excellent discrimination abilities, with curves that closely approached the top-left corner of the plot. This indicates a high true positive rate and a low false positive rate across a range of classification thresholds. These models demonstrated superior performance in distinguishing between fraudulent and non-fraudulent transactions.
