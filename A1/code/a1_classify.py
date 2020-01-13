import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

# Added libraries
import sys
import numpy as np
from sklearn.feature_selection import chi2


# (Reference on confusion matrix: https://www.python-course.eu/confusion_matrix.php)

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    correct = C.trace() # diagnol terms are correctly predicted ones
    total = C.sum()
    return 0. if total==0. else correct/total


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    correct = C.diagonal()
    total_per_row = np.sum(C, axis=1)
    recall = np.zeros(correct.shape)
    for i, total in np.ndenumerate(total_per_row):
        if total != 0.:
            recall[i] = correct[i]/total
    return recall


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    correct = C.diagonal()
    total_per_col = np.sum(C, axis=0)
    precision = np.zeros(correct.shape)
    for i, total in np.ndenumerate(total_per_col):
        if total != 0.:
            precision[i] = correct[i]/total
    return precision



def evaluate(i, C, output_dir):
    """Compute, print, and write the resultes into the csv file

    Arguments:
        i {int} -- index of model
        C {Numpy array} -- the confusion matrix
        output_dir {string} -- path of directory to write output to

    Returns:
        accuracy {float} -- the accuracy, used to find the best model
    """
    classifier_name = MODELS[i]
    acc, rec, prec = accuracy(C), recall(C), precision(C)

    # Print
    print("Model {}. {}:\nAccuracy: {}\nAverage recall: {}\nAverage precision: {}\nConfusion matrix:\n{}".format(
        i, classifier_name, acc, rec.mean(), prec.mean(), C))

    # Txt log
    with open(f"{output_dir}/a1_3.1.txt", "a+") as outf:
        # For each classifier, compute results and write the following output:
        outf.write(f'Results for {classifier_name}:\n')  # Classifier name
        outf.write(f'\tAccuracy: {acc:.4f}\n')
        outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
        outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
        outf.write(f'\tConfusion Matrix: \n{C}\n\n')
    return acc

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    '''
    print('\nProcessing Section 3.1...')
    global MODELS
    MODELS = {1: "SGDClassifier",
              2: "GaussianNB",
              3: "RandomForestClassifier",
              4: "MLPClassifier",
              5: "AdaBoostClassifier"}
    accuracies = np.zeros((5,))
    index = 1

    # Start training
    print("+++++ {}. +++++++++++++++++++++++++++++++++++".format(index))
    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    C = confusion_matrix(y_test, y_pred)
    accuracies[index-1] = evaluate(index, C, output_dir)
    index += 1

    print("+++++ {}. +++++++++++++++++++++++++++++++++++".format(index))
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    C = confusion_matrix(y_test, y_pred)
    accuracies[index-1] = evaluate(index, C, output_dir)
    index += 1

    print("+++++ {}. +++++++++++++++++++++++++++++++++++".format(index))
    clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    C = confusion_matrix(y_test, y_pred)
    accuracies[index-1] = evaluate(index, C, output_dir)
    index += 1

    print("+++++ {}. +++++++++++++++++++++++++++++++++++".format(index))
    clf = MLPClassifier(alpha=0.05)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    C = confusion_matrix(y_test, y_pred)
    accuracies[index-1] = evaluate(index, C, output_dir)
    index += 1

    print("+++++ {}. +++++++++++++++++++++++++++++++++++".format(index))
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    C = confusion_matrix(y_test, y_pred)
    accuracies[index-1] = evaluate(index, C, output_dir)
    index += 1

    # Return the model index giving the best results
    iBest = np.argmax(accuracies) + 1
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('\nProcessing Section 3.2...')

    if iBest == 1:
        clf = LinearSVC()
    elif iBest == 2:
        clf = SVC(gamma=2, max_iter=1000)
    elif iBest == 3:
        clf = RandomForestClassifier(n_estimators=10, max_depth=5)
    elif iBest == 4:
        clf = MLPClassifier(alpha=0.05)
    elif iBest == 5:
        clf = AdaBoostClassifier()

    data_sizes = [1000, 5000, 10000, 15000, 20000]
    accuracies = np.zeros((5,))

    with open(f"{output_dir}/a1_3.2.txt", "a+") as outf:

        for i, data_size in enumerate(data_sizes):
            X_train_mini, y_train_mini = X_train[:data_size], y_train[:data_size]

            clf.fit(X_train_mini, y_train_mini)
            y_pred = clf.predict(X_test)
            C = confusion_matrix(y_test, y_pred)
            accuracies[i] = accuracy(C)

            # For each number of training examples, compute results and write the following output:
            outf.write(f'{data_size}: {accuracies[i]:.4f}\n')

    X_1k, y_1k = X_train[:1000], y_train[:1000]
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('Processing Section 3.3...')

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:

        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        # for each number of features k_feat, write the p-values for
        # that number of features:
            # outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}')

        # outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        # outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        # outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        # outf.write(f'Top-5 at higher: {top_5}\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    print('TODO Section 3.4')

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # print("Testing confusion matrix computations+++++++++")
    # C_test = np.array(
    #     [[5825,    1,   49,   23,    7,   46,   30,   12,   21,   26],
    #     [1, 6654,   48,   25,   10,   32,   19,   62,  111,   10],
    #     [2,   20, 5561,   69,   13,   10,    2,   45,   18,    2],
    #     [6,   26,   99, 5786,    5,  111,    1,   41,  110,   79],
    #     [4,   10,   43,    6, 5533,   32,   11,   53,   34,   79],
    #     [3,    1,    2,   56,    0, 4954,   23,    0,   12,    5],
    #     [31,    4,   42,   22,   45,  103, 5806,    3,   34,    3],
    #     [0,    4,   30,   29,    5,    6,    0, 5817,    2,   28],
    #     [35,    6,   63,   58,    8,   59,   26,   13, 5394,   24],
    #     [16,   16,   21,   57,  216,   68,    0,  219,  115, 5693]])
    # print(recall(C_test).mean())
    # print("++++++++++++++++++++++++++++++++++++++++++++++")




    # TODO: load data and split into train and test.
    np.random.seed(999)
    input_file, output_dir = args.input, args.output_dir
    npz = np.load(input_file)
    feats = npz[npz.files[0]]

    # X: the first 173 columns in the feature array, y: the last one
    X, y = feats[:, :-1], feats[:, -1]
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2)

    # TODO : complete each classification experiment, in sequence.
    # Create/clean up the files
    open(f"{output_dir}/a1_3.1.txt", "w+").close()
    open(f"{output_dir}/a1_3.2.txt", "w+").close()
    open(f"{output_dir}/a1_3.3.txt", "w+").close()
    # iBest = class31(output_dir, X_train, X_test, y_train, y_test)
    iBest = 5
    X_1k, y_1k = class32(output_dir, X_train, X_test, y_train, y_test, iBest)
    # class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    # class34(filename, iBest)

    # python3 a1_classify.py -i feats_medium.npz -o classifier_output_mini
    # python3 a1_classify.py -i feats.npz -o classifier_output
