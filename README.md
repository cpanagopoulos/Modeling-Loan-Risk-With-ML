# Using Machine Learning to Evaluate Loan Applicant Risk
 
![Credit Risk](Images/credit-risk.jpg)

## Background

Built and evaluated several machine learning models to predict credit risk using data typically seen from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem. For example, the number of good loans is much larger than the number of at-risk loans. We used different techniques for training and evaluating models with imbalanced classes. We used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

- - -
## Resampling

Used the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data. Built and evaluated logistic regression classifiers using the resampled data.

1. Read the CSV into a DataFrame.

2. Split data into Training and Testing sets.

3. Scaled the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.

4. Used the provided code to run a Simple Logistic Regression:
    * Fit the `logistic regression classifier`.
    * Calculate the `balanced accuracy score`.
    * Display the `confusion matrix`.
    * Print the `imbalanced classification report`.

## Oversampled Algorithmic Data 

1. Oversampled the data using the `Naive Random Oversampler` and `SMOTE` algorithms.

2. Undersampled the data using the `Cluster Centroids` algorithm.

3. Over- and undersampled using a combination `SMOTEENN` algorithm.

## Training Resampled Data

1. Trained a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.

2. Calculated the `balanced accuracy score` from `sklearn.metrics`.

3. Displayed the `confusion matrix` from `sklearn.metrics`.

4. Printed the `imbalanced classification report` from `imblearn.metrics`.

Used the above to answer the following questions:

* Which model had the best balanced accuracy score?

> The oversampling model had the best balanced accuracy score.

* Which model had the best recall score?

> It was a tie across the models because they all scored 99%.

* Which model had the best geometric mean score?

*  The oversampling, SMOTE, and combined models all had a score of 99%.

## Ensemble Learning

In this section, we trained and compared two different ensemble classifiers to predict loan risk and evaluate each model. We used the [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) and the [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html).

## Project Review Questions

* Which model had the best balanced accuracy score?

> The Easy Ensemble model had the best balanced accuracy score.

* Which model had the best recall score?

> Both models scored a recall score of .74.

* Which model had the best geometric mean score?

> Both models scored .82.
