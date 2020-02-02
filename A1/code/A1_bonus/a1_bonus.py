# The very basics
import argparse
import os
import sys
import timeit
import numpy as np
import pandas as pd
import csv
import json

# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Pre-process
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit


# Post-process
from scipy import stats
from scipy.stats import ttest_rel
from sklearn.metrics import confusion_matrix


#########################################
######### Code from a1_classify #########
#########################################

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    correct = C.trace()  # diagnol terms are correctly predicted ones
    total = C.sum()
    return 0. if total == 0. else correct/total


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


#########################################
######### Other helper functions ########
#########################################
def model_selection(i, param=None):
    """Return the model accurding to the model index given


    Arguments:
        i {int} -- integer starting from 1

    Keyword Arguments:
        param {anything} -- we will only need 1 param for any one classifier anyways
        Note : d=5, alpha=0.05, lr=1, mf=None
    Returns:
        function -- the clf model
    """

    if i == 1:
        return SGDClassifier()
    elif i == 2:
        return GaussianNB()
    elif i == 3:
        return RandomForestClassifier(n_estimators=10, max_depth=param)
    elif i == 4:
        return MLPClassifier(alpha=param)
    elif i == 5:
        return AdaBoostClassifier(learning_rate=param)
    elif i == 6:
        return SVC(kernel='linear', max_iter=10000)
    elif i == 7:
        return SVC(kernel='rbf', max_iter=10000, gamma=2)
    elif i == 8:
        return DecisionTreeClassifier(random_state=0, max_features=param)
    else:
        print("i must be an integer between 1 and 5, but input is", iBest)
        return None


def train_evaluate(X_train, X_test, y_train, y_test, i, param=None, output_dir="bonus_output", feat_num=10, full_output=True):
    """Train, test, and evaluate the results

    Arguments:
        X_train, X_test, y_train, y_test -- just pass them from main
        i {[type]} -- model index, starts from 1, not 0!

    Keyword Arguments:
        output_dir {str} -- output folder name (default: {"bonus_output"})
        full_output {bool} -- Determines if we output only accuracy or other stats as well (default: {True})
        feat_num {int} -- If >0, then it's the number of featrues selected, otherwise not feature-based (default: {10})

    Returns:
        [type] -- [description]
    """
    # Select model
    classifier_name = MODELS[i]
    clf = model_selection(i, param)

    # Train and predict
    if feat_num > 0:
        selector = SelectKBest(f_classif, k=feat_num)
        X_train_new = selector.fit_transform(X_train, y_train)
        X_test_idx = selector.get_support(indices=True)
        X_test_new = X_test[:, X_test_idx]
        # pp = selector.pvalues_[X_test_idx]
        clf.fit(X_train_new, y_train)
        y_pred = clf.predict(X_test_new)

    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)


    C = confusion_matrix(y_test, y_pred)
    acc, rec, prec = accuracy(C), recall(C), precision(C)

    # Print
    print("Classifier: {}\nNumber of features: {}\nParameter: {}".format(
        classifier_name, feat_num, param))
    print("Accuracy: {}\nAverage recall: {}\nAverage precision: {}\nConfusion matrix:\n{}".format(
        acc, rec.mean(), prec.mean(), C))

    # Txt log
    if full_output:
        with open(f"{output_dir}/a1_bonus_classifiers.txt", "a+") as outf:
            # For each classifier, compute results and write the following output:
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'Feature-based: {feat_num>0}\n')
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{C}\n\n')
    return acc


#########################################
######### Code from a1_classify #########
#########################################
def classifers(X_train, X_test, y_train, y_test, output_dir):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    '''
    with open(f"{output_dir}/a1_bonus_classifiers.txt", "a+") as outf:
        # For each classifier, compute results and write the following output:
        outf.write(f'##############################################\n')
        outf.write(f'Testing performance with different classifiers\n')
        outf.write(f'##############################################\n')

    print('\nProcessing Section 3.1...')

    # TESTS = {1: [0]} # small sample tester

    accuracies = {}
    for key in range(1, len(TESTS)+1):
        model = MODELS[key]
        accuracies[model] = {}
        for param in TESTS[key]:
            print("+++++ Model {} with param {} +++++++++++++++++++++++++++++++++++".format(key, param))
            start_timer = timeit.default_timer()
            accuracies[model][param] = (
                train_evaluate(
                    X_train, X_test, y_train, y_test, key, param, feat_num=0),
                train_evaluate(
                    X_train, X_test, y_train, y_test, key, param, feat_num=20)
            )
            stop_timer = timeit.default_timer()
            print("+++++ Finished in {} seconds +++++\n".format(stop_timer - start_timer))

    df = pd.DataFrame.from_dict(accuracies)
    print(df.to_string())

    with open(f"{output_dir}/a1_bonus_classifiers.txt", "a+") as outf:
        # For each classifier, compute results and write the following output:
        outf.write(f'All results:\n{df.to_string()}\n')  # Classifier name
        outf.write(f'##############################################\n\n\n')
    return


#########################################
######### Code for new features #########
#########################################
def get_new_feats(feats_old, data):
    """Use the old features extracted from Task 2 and the raw data from Task 2 to get new features

    Returns:
        ndarray -- size should be (len(feats.old), size of new features (>=174))
    """
    print("The old features' dimension is {}".format(feats_old.shape))
    feats_new = np.hstack((feats_old, np.zeros((len(data), 4))))

    for i in range(feats_new.shape[0]):
        feats_new[i][173] = data[i]['controversiality']
        feats_new[i][174] = data[i]['score']
        feats_new[i][176] = data[i]['body'].count('obama')
        feats_new[i][175] = data[i]['body'].count('clinton')

    print("Finishes adding new features, now the dimension is {}".format(feats_new.shape))
    return feats_new


def features(X_train, X_test, y_train, y_test, output_dir, clf, key=None, param=None, new=False):
    """[summary]
    
    Arguments:
        X_train {[type]} -- [description]
        X_test {[type]} -- [description]
        y_train {[type]} -- [description]
        y_test {[type]} -- [description]
        output_dir {[type]} -- [description]
    """
    if new:
        label = "new"
        new_num_feats = X_train.shape[1]
    else:
        label = "old"
        new_num_feats = 0
        
    print("\n########################## Testing {} features ##########################\n".format(label))
    with open(f"{output_dir}/a1_bonus_classifiers.txt", "a+") as outf:
        # For each classifier, compute results and write the following output:
        outf.write(f'##############################################\n')
        outf.write(f'Testing performance with {label} features\n')
        outf.write(f'##############################################\n')
    
    # If we passed in the new features, we want to know their p-values
    if new:
        print("+++++ Checking p value ++++++ ")
        k = new_num_feats - 173
        selector = SelectKBest(f_classif, k)
        X_new = selector.fit_transform(X_train, y_train)
        
        pp = selector.pvalues_
        p_values_new = pp[173:new_num_feats]

        pp_idx = selector.get_support(indices=True)
        p_values_k = pp[pp_idx]
        
        print("The p-values of new features are: \t{},\nCompared to the best {} p-values: \t{}".format(p_values_new, k, p_values_k))
        print("Also compared to the worst p-value: \t{}".format(np.nanmax(pp)))

        # TODO: Check Piazza
        with open(f"{output_dir}/a1_bonus_features.txt", "a+") as outf:
            outf.write(
                f'Best features\' p-values: {[pval for pval in p_values_k]}\n')
            outf.write(
                f'New features \' p-values: {[pval for pval in p_values_new]}\n')

    # Now we check the accuracies with either the new or old
    print("\n+++++ Checking accuracies with best {} features ++++++ ".format(20))
    selector = SelectKBest(f_classif, k=20)

    acc1K = train_evaluate(
        X_train[:1000], X_test, y_train[:1000], y_test, key, param, feat_num=20, full_output=False)
    print('--------------------------------')
    acc = train_evaluate(
        X_train, X_test, y_train, y_test, key, param, feat_num=20, full_output=False)
    # print("\tAccuracy with 1K data:", acc1K)
    # print("\tAccuracy with full data:", acc)
    
    with open(f"{output_dir}/a1_bonus_features.txt", "a+") as outf:
            outf.write(
                f'\nAccuracy with 1K data and top 20 features, {label}: {acc1K}\n')
            outf.write(
                f'Accuracy with full data and top 20 features, {label}: {acc}\n')
            
    ######################################################################
    # Now we check the accuracies without using 
    print("\n+++++ Checking accuracies with all features ++++++ ")

    acc1K = train_evaluate(
        X_train[:1000], X_test, y_train[:1000], y_test, key, param, feat_num=0, full_output=False)
    print('--------------------------------')
    acc = train_evaluate(
        X_train, X_test, y_train, y_test, key, param, feat_num=0, full_output=False)
    # print("\tAccuracy with 1K data:", acc1K)
    # print("\tAccuracy with full data:", acc)

    with open(f"{output_dir}/a1_bonus_features.txt", "a+") as outf:
                outf.write(
                    f'\nAccuracy with 1K data with all features, {label}: {acc1K}\n')
                outf.write(
                    f'Accuracy with full data with all features, {label}: {acc}\n')
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i2", "--input2", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-i1", "--input1", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write bonus files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # Mute warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    # Load data and split into train and test
    np.random.seed(999)
    input_file, output_dir = args.input2, args.output_dir
    npz = np.load(input_file)
    feats = npz[npz.files[0]]

    # X: the first 173 columns in the feature array, y: the last one
    X, y = feats[:, :-1], feats[:, -1]
    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.2)


    ##########################################################
    ######### Test and analyze different classifiers #########
    ##########################################################
    global MODELS, TESTS
    MODELS = {1: "SGDClassifier",
              2: "GaussianNB",
              3: "RandomForestClassifier",
              4: "MLPClassifier",
              5: "AdaBoostClassifier",
              6: "Linear SVC",
              7: "Gaussian SVC",
              8: "DecisionTreeClassifier"}

    TESTS = {1: [0],
             2: [0],
             3: [1, 5, 10, 15],  # max depths in RandomForestClassifier
             4: [0.01, 0.05, 0.1, 0.5, 1],  # alphas in MLPClassifier
             # learning rate in AdaBoostClassifier
             5: [0.05, 0.1, 1.0, 1.5, 2.0],
             6: [0],
             7: [0],
             8: [None, 'sqrt', 'log2']}  # max_features in DecisionTreeClassifier

    # open(f"{output_dir}/a1_bonus_classifiers.txt", "w+").close()
    # classifers(X_train, X_test, y_train, y_test, output_dir) # TODO: activate this before submission

    ##########################################################
    ######### Process and test new features ##################
    ##########################################################
    # Model used: MLPClassifier with alpha = 0.01
    i, param = 4, 0.01  # chosen from the feature-based ones, as we are considering features now
    clf = model_selection(i, param)
    
    # Get new features
    data = json.load(open(args.input1))
    X_new = get_new_feats(X, data)
    (X_train, X_test, X_train_new, X_test_new, y_train, y_test) = train_test_split(
        X, X_new, y, test_size=0.2)

    open(f"{output_dir}/a1_bonus_features.txt", "w+").close()
    features(X_train_new, X_test_new, y_train, y_test,
             output_dir, clf, i, param, new=True)
    features(X_train, X_test, y_train, y_test,
             output_dir, clf, i, param, new=False)


    # python3 a1_bonus.py -i1 preproc.json -i2 feats.npz -o bonus_output
    # python3 a1_bonus.py -i1 preproc_bonus.json -i2 feats.npz -o bonus_output
