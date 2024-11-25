"""
CS 471 Final Project: Fraud Detection using Machine Learning
Team 2: Shane Dyrdahl, Alastair Raymond, Riley Saliba, Dax Taraleskof
Date: November 2024

Fraud detection using machine learning models.
We preprocesses the Fraud dataset, address class imbalance with SMOTE, and evaluate
the performance of Decision Tree and Random Forest classifiers. The models are tuned
using GridSearchCV, and evaluation metrics include recall, precision, and F1-score.

Key Features:
- Preprocessing with encoding and oversampling.
- Implementation of rules-based and machine learning approaches.
- Model evaluation using confusion matrices and classification reports.
"""

import pickle  # Used for saving and loading datasets for efficiency.
from imblearn.over_sampling import SMOTE  # For oversampling in imbalanced datasets.
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing   # Tools for encoding categorical variables.
#from matplotlib import pyplot  # For data visualization.
import pandas       # Used for handling tabular data.

# csv format:
# step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
# 6,362,620 total rows

# Number of rows in the dataset (for reference purposes).
row_count = 6362620

dataset_path = '/home/eel52/dataset'  # Path to the pickled dataset file.

# Load the dataset from a pickle file or fall back to reading the CSV file.
try:
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
except FileNotFoundError:
    dataset = pandas.read_csv('Fraud.csv')  # Load dataset from CSV if pickle not found.
    with open(dataset_path, "wb") as fp:
        pickle.dump(dataset, fp)  # Save dataset to a pickle for future use.

# Visualization example (currently commented out for performance reasons):
# Uncomment to generate a scatter plot of 'isFraud' (x-axis) vs. 'amount' (y-axis).
# pyplot.scatter(dataset[['isFraud']], dataset[['amount']])
# pyplot.show()

# Encode categorical variables to numeric values for compatibility with ML models.
label_encoder = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder()
dataset['type'] = label_encoder.fit_transform(dataset['type'])  # Encode transaction type.
dataset['nameOrig'] = label_encoder.fit_transform(dataset['nameOrig'])  # Encode origin account ID.
dataset['nameDest'] = label_encoder.fit_transform(dataset['nameDest'])  # Encode destination account ID.

# Define the features (X) and the target variable (y).
X = dataset[
    ['type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']]
y = dataset[['isFraud']]

# Split the dataset into training and testing sets (80/20 split).
training_X, testing_X, training_y, testing_y = train_test_split(X, y, test_size=0.20)

# Further split the training data into training and evaluation sets (60/20 split).
training_X, evaluation_X, training_y, evaluation_y = train_test_split(training_X, training_y, test_size=0.20)

# Use SMOTE to oversample the minority class and balance the training data.
temp_X, temp_y = SMOTE(random_state=130).fit_resample(training_X, training_y)
training_X = temp_X
training_y = temp_y

# Function to evaluate the rules-based model.
def evaluate(threshold, data, actual):
    """
    Evaluate a rules-based model for fraud detection.

    Parameters:
        threshold (float): Transaction amount threshold for fraud detection.
        data (DataFrame): Dataset containing transaction features.
        actual (Series): True labels for fraud (1) or non-fraud (0).

    Outputs:
        Prints recall, precision, and F1-score for the rules-based model.
    """
    
    false_negative_count = 0
    true_positive_count = 0
    merged_data = data.copy()
    merged_data.insert(8, "isFraud", actual, True)
    
    # Identify potential fraudulent transactions based on the threshold.
    potential_fraud = merged_data.loc[(merged_data['amount'] < threshold)]
    actual_fraud = potential_fraud.loc[(potential_fraud['isFraud'] == 1)]
    not_fraud = potential_fraud.loc[(potential_fraud['isFraud'] == 0)]
    
    # Calculate recall, precision, and F1-score.
    true_positive_count = len(actual_fraud)
    total_fraud_count = len(merged_data.loc[(merged_data['isFraud'] == 1)])
    false_negative_count = total_fraud_count - len(actual_fraud)
    false_positive_count = len(not_fraud)
    
    # Calculate recall metric
    recall = (true_positive_count / (true_positive_count + false_negative_count))
    print("Recall: " + str(recall))
    
    # Calculate precision metric
    precision = (true_positive_count / (true_positive_count + false_positive_count))
    print("Precision: " + str(precision))
    
    # Calculate F1-score
    print("F1-Score: " + str(2 * ((precision * recall) / (precision + recall))))


# evaluate(10000000, evaluation_X, evaluation_y)


# == training the model ==

# Function to evaluate Decision Tree models.
def dt_evaluate(X_dataset, y_dataset, grid_search):
    """
    Evaluate a Decision Tree model using GridSearchCV results.

    Parameters:
        X_dataset (DataFrame): Features for evaluation/testing.
        y_dataset (Series): True labels for evaluation/testing.
        grid_search (GridSearchCV): Trained GridSearchCV object.
    """
    
    # get the best tuned parameters and the score, which should be recall
    best_params = grid_search.best_params_  # Retrieve the best parameters.
    best_score = grid_search.best_score_  # Retrieve the best recall score.
    
    print(f"Best params: {best_params}")
    print(f"Best recall from GridSearchCV tuning: {best_score * 100:.2f}%")
    
    # Evaluate the best estimator of the decision tree with the tuned hyperparams
    best_model = grid_search.best_estimator_
    testing_predictions = best_model.predict(X_dataset)
    testing_recall = recall_score(y_dataset, testing_predictions)
    print(f"recall of best estimator from GridSearchCV on test set for DT: {testing_recall * 100:.2f}%")
    
    # Print classification report
    print(classification_report(y_dataset, testing_predictions, target_names=['isNotFraud', 'isFraud']))
    
    # Print confusion matrix
    print(f"Confusion matrix (DT):\n " + str(confusion_matrix(y_dataset, testing_predictions)))


# Function to evaluate Random Forest models.
def rf_evaluate(X_dataset, y_dataset, grid_search):
    """
    Evaluate a Random Forest model using GridSearchCV results.

    Parameters:
        X_dataset (DataFrame): Features for evaluation/testing.
        y_dataset (Series): True labels for evaluation/testing.
        grid_search (GridSearchCV): Trained GridSearchCV object.
    """
    
    # Get the best tuned parameters and the score, which should be recall
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best params: {best_params}")
    print(f"Best recall from GridSearchCV: {best_score * 100:.2f}%")
    
    # Evaluate the best estimator of the decision tree with the tuned hyperparams
    best_model = grid_search.best_estimator_
    testing_predictions = best_model.predict(X_dataset)
    testing_recall = recall_score(y_dataset, testing_predictions)
    print(f"recall of best estimator from GridSearchCV on test set for RF: {testing_recall * 100:.2f}%")
    
    # Print classification report
    print(classification_report(y_dataset, testing_predictions, target_names=['isNotFraud', 'isFraud']))
    
    # Print confusion matrix
    print(f"Confusion matrix (RF):\n " + str(confusion_matrix(y_dataset, testing_predictions)))


# Define hyperparameters for Decision Tree tuning.
param_grid_dt = {
    'max_depth': [4],
    'min_samples_split': [2]
}

# Train Decision Tree with GridSearchCV.
# Tune the hyperparams and then fit the training data on it
decision_tree = DecisionTreeClassifier(random_state=130)
grid_search_dt = GridSearchCV(estimator=decision_tree, param_grid=param_grid_dt, cv=5, scoring='recall', n_jobs=-1,
                              verbose=1)
grid_search_dt.fit(training_X, training_y)

# Evaluate on evaluation dataset and tune hyperparameters based on result
dt_evaluate(evaluation_X, evaluation_y, grid_search_dt)

# Perform final testing on testing dataset
dt_evaluate(testing_X, testing_y, grid_search_dt)

# Define hyperparameters for Random Forest tuning.
# using best params from FirstTrainingRF.png
param_grid_rf = {
    'n_estimators': [25],  # Number of trees in the forest.
    'max_depth': [2],  # Maximum tree depth.
    'min_samples_leaf': [1]  # Minimum samples at leaf nodes.
}

# Train Random Forest with GridSearchCV.
random_forest = RandomForestClassifier(class_weight='balanced', random_state=130)
grid_search_rf = GridSearchCV(estimator=random_forest, param_grid=param_grid_rf, scoring='recall', n_jobs=-1, verbose=1)
# random forest wants a 1d array, hence the ravel()
grid_search_rf.fit(training_X, training_y.values.ravel())

# Evaluate the Random Forest model on evaluation and testing sets.
rf_evaluate(evaluation_X, evaluation_y, grid_search_rf)
rf_evaluate(testing_X, testing_y, grid_search_rf)
