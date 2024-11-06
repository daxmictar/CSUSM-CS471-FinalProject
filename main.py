import csv
import statistics
import matplotlib.pyplot as plt
import pickle

# csv format:
# step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
# 6,362,620 total rows

row_count = 6362620
dataset = []

dataset_path = '/home/eel52/dataset'

# read data from list file or from csv
try:
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
except:
    with open('Fraud.csv', mode = 'r') as file:
        csv_reader = csv.reader(file)

        line_count = 0
        for line in csv_reader:
            print("Progress: " + str(int((line_count / row_count) * 100)) + "% ", end = '\r')     # ugly
            if (line_count > 0):
                new_entry = []
                for i in range(10):
                    new_entry.append(line[i])
                dataset.append(new_entry)

            line_count = line_count + 1

        with open(dataset_path, "wb") as fp:
            pickle.dump(dataset, fp)


amount_array = []
for line in dataset:
    amount_array.append(float(line[2]))

# visualize data

#plt.boxplot(amount_array, sym='o')
#plt.show()

median = statistics.median(amount_array)

print("Median = " + str(median))

#standard_deviation = statistics.stdev(amount_array)

#print("Standard deviation = " + str(standard_deviation))

#average = statistics.mean(amount_array)

#print("Average = " + str(average))

def evaluate(threshold):
    false_negative_count = 0
    true_positive_count = 0
    for line in dataset:
        try:
            # possible fraud detected
    #        if ((abs(float(line[2]) - float(median)) > threshold) and (line[1] == "TRANSFER" or line[1] == "CASH_OUT")):
            if ((float(line[2]) - float(median)) > threshold):
                if (int(line[9]) == 0):
                    false_negative_count = false_negative_count + 1
                if (int(line[9]) == 1):
                    true_positive_count = true_positive_count + 1
        except:
            print()

    # calculate recall metric
    print("Recall: " + str(true_positive_count / (true_positive_count + false_negative_count)))


evaluate(1000000)
evaluate(2000000)
evaluate(3000000)
evaluate(4000000)
evaluate(5000000)
evaluate(6000000)
evaluate(7000000)
