# Using Machine Learning to Evaluate Loan Applicant Risk
 
![Credit Risk](Images/credit-risk.jpg)

## Background

Built and evaluated several machine learning models to predict credit risk using data typically seen from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem. For example, the number of good loans is much larger than the number of at-risk loans. We used different techniques for training and evaluating models with imbalanced classes. We used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

- - -
### Resampling

Use the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data and build and evaluate logistic regression classifiers using the resampled data.

To begin:

1. Read the CSV into a DataFrame.

2. Split the data into Training and Testing sets.

3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.

4. Use the provided code to run a Simple Logistic Regression:
    * Fit the `logistic regression classifier`.
    * Calculate the `balanced accuracy score`.
    * Display the `confusion matrix`.
    * Print the `imbalanced classification report`.

Next you will:

1. Oversample the data using the `Naive Random Oversampler` and `SMOTE` algorithms.

2. Undersample the data using the `Cluster Centroids` algorithm.

3. Over- and undersample using a combination `SMOTEENN` algorithm.


For each of the above, you will need to:

1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.

2. Calculate the `balanced accuracy score` from `sklearn.metrics`.

3. Display the `confusion matrix` from `sklearn.metrics`.

4. Print the `imbalanced classification report` from `imblearn.metrics`.


Use the above to answer the following questions:

* Which model had the best balanced accuracy score?
>
* Which model had the best recall score?
>
* Which model had the best geometric mean score?

### Ensemble Learning

In this section, you will train and compare two different ensemble classifiers to predict loan risk and evaluate each model. You will use the [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) and the [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html). Refer to the documentation for each of these to read about the models and see examples of the code.

To begin:

1. Read the data into a DataFrame using the provided starter code.

2. Split the data into training and testing sets.

3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.


Then, complete the following steps for each model:

1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.

2. Calculate the balanced accuracy score from `sklearn.metrics`.

3. Display the confusion matrix from `sklearn.metrics`.

4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.

5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.


Use the above to answer the following questions:

* Which model had the best balanced accuracy score?

* Which model had the best recall score?

* Which model had the best geometric mean score?

* What are the top three features?

- - -
