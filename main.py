import pickle
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
median = float(training_X[['amount']].median().iloc[0])

print("Median = " + str(median))

def evaluate(threshold, data, actual):
    false_negative_count = 0
    true_positive_count = 0
    merged_data = data.copy()
    merged_data.insert(8, "isFraud", actual, True)
    false_negative_count = len(merged_data.loc[((merged_data['amount'] - (median)) <= threshold) & ((merged_data['isFraud']) == 1)])
    true_positive_count = len(merged_data.loc[((merged_data['amount'] - (median)) > threshold) & ((merged_data['isFraud']) == 1)])

    # calculate recall metric
    print("Recall: " + str(true_positive_count / (true_positive_count + false_negative_count)))


evaluate(1000000, evaluation_X, evaluation_y)
evaluate(2000000, evaluation_X, evaluation_y)
evaluate(3000000, evaluation_X, evaluation_y)
evaluate(4000000, evaluation_X, evaluation_y)
evaluate(5000000, evaluation_X, evaluation_y)
evaluate(6000000, evaluation_X, evaluation_y)
evaluate(7000000, evaluation_X, evaluation_y)
