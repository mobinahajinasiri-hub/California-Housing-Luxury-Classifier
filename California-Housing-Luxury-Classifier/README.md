Housing Price Classification Project 🏡
Overview

This project performs binary classification on the California housing dataset. The goal is to predict whether a house’s value is above the 75th percentile or not.

Dataset

File: housing.csv

Contains numerical and categorical features related to houses and neighborhoods.

Missing values have been removed.

Target Variable

median_house_value is the target.

Binary label created:

1 → house value ≥ 75th percentile

0 → house value < 75th percentile

Features

All features except median_house_value and the label are used.

Categorical features are one-hot encoded using pd.get_dummies().

Methodology

Data preprocessing: Handle missing values, encode categorical features.

Train-test split: 80% training, 20% testing (stratified by label).

Model: Logistic Regression (max_iter=1000)

Evaluation metrics: Precision, Recall, ROC-AUC

Visualization: ROC curve plotted to show model performance.

Results

After training the Logistic Regression model:

Precision: 0.80

Recall: 0.65

AUC: 0.91

⚠️ Note: Scaling the data using StandardScaler may improve model convergence and slightly increase performance. Alternative solver options are also available in the Logistic Regression documentation
.

How to Run

Install required packages:

pip install pandas numpy scikit-learn matplotlib


Place housing.csv in the project directory.

Run the script:

python housing.py