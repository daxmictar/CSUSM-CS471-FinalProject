import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from matplotlib import pyplot
import pandas

# csv format:
# step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
# 6,362,620 total rows

row_count = 6362620

dataset_path = '/home/eel52/dataset'

# read data from list file or from csv
try:
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
except:
    dataset = pandas.read_csv('Fraud.csv')
    with open(dataset_path, "wb") as fp:
        pickle.dump(dataset, fp)


# visualization (takes a while)
#pyplot.scatter(dataset[['isFraud']], dataset[['amount']])
#pyplot.show()

# encode strings
label_encoder = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder()
dataset['type'] = label_encoder.fit_transform(dataset['type'])
dataset['nameOrig'] = label_encoder.fit_transform(dataset['nameOrig'])
dataset['nameDest'] = label_encoder.fit_transform(dataset['nameDest'])

# split dataset into X and y
X = dataset[['type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']]
y = dataset[['isFraud']]

# split dataset into training and testing
training_X, testing_X, training_y, testing_y = train_test_split(X, y, test_size = 0.20)

# oversample training dataset
temp_X, temp_y = SMOTE(random_state=130).fit_resample(training_X, training_y)
training_X = temp_X
training_y = temp_y

# split training dataset into training and evaluation
training_X, evaluation_X, training_y, evaluation_y = train_test_split(training_X, training_y, test_size = 0.20)

# calculate statistics
#median = float(training_X[['amount']].median().iloc[0])

#print("Median = " + str(median))

def evaluate(threshold, data, actual):
    false_negative_count = 0
    true_positive_count = 0
    merged_data = data.copy()
    merged_data.insert(8, "isFraud", actual, True)
    potential_fraud = merged_data.loc[(merged_data['amount'] < threshold)]
    actual_fraud = potential_fraud.loc[(potential_fraud['isFraud'] == 1)]
    not_fraud = potential_fraud.loc[(potential_fraud['isFraud'] == 0)]
    true_positive_count = len(actual_fraud)
    total_fraud_count = len(merged_data.loc[(merged_data['isFraud'] == 1)])
    false_negative_count = total_fraud_count - len(actual_fraud)
    false_positive_count = len(not_fraud)

    # calculate recall metric
    recall = (true_positive_count / (true_positive_count + false_negative_count))
    print("Recall: " + str(recall))
    
    # calculate precision metric
    precision = (true_positive_count / (true_positive_count + false_positive_count))
    print("Precision: " + str(precision))

    # calculate F1-score
    print("F1-Score: " + str(2 * ((precision * recall) / (precision + recall))))

evaluate(10000000, evaluation_X, evaluation_y)


# == training the model ==

def dt_evaluate(X_dataset, y_dataset, grid_search):
    # get the best tuned parameters and the score, which should be recall
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"best params: {best_params}")
    print(f"best recall from gridsearch tuning: {best_score*100:.2f}%")

    # evaluate the best estimator of the decision tree with the tuned hyperparams 
    best_model = grid_search.best_estimator_
    testing_predictions = best_model.predict(X_dataset)
    testing_recall = recall_score(y_dataset, testing_predictions)
    print(f"recall of best estimator from GridSearchCV on test set: {testing_recall*100:.2f}%")

    # print classification report
    print(classification_report(y_dataset, testing_predictions, target_names=['isNotFraud', 'isFraud']))

    # print confusion matrix
    print(f"Confusion matrix:\n " + str(confusion_matrix(y_dataset, testing_predictions)))

# hyperparams to tune
param_grid = {
#    'max_depth': [i for i in range(2, 5)],
    'max_depth': [4],
#    'min_samples_split': [i for i in range(2, 5)]
    'min_samples_split': [2]
}

# tune the hyperparams and then fit the training data on it
decision_tree = DecisionTreeClassifier(random_state=130)
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)
grid_search.fit(training_X, training_y)

# evaluate on evaluation dataset and tune hyperparameters based on result
dt_evaluate(evaluation_X, evaluation_y, grid_search)

# perform final testing on testing dataset
dt_evaluate(testing_X, testing_y, grid_search)
