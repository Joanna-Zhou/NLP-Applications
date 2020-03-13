from a3_gmm import *
from sklearn.decomposition import PCA
import time

dataDir = '/u/cs401/A3/data/'
# dataDir = '/Users/joanna.zyz/NLP-Applications/A3/data'


def fitPCA(d, dd):
    ''' Fit and save the PCA-transformed input data X
        Convention: X (dim = d) => reduced => XX (dim = dd)
        For processing speed, also return a dictionary of X collected per-speaker
    '''
    x_list = {} # a dict of X per speaker
    X = np.empty((0, d)) # an array for all speakers

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # print("walking in dirs")
            files = fnmatch.filter(os.listdir(
                os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            x = np.empty((0, d)) # For this speaker only
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                x = np.append(x, myMFCC, axis=0)
            x_list[speaker] = x
            X = np.append(X, x, axis=0)

    pca = PCA(n_components=dd)
    pca.fit(X)
    np.save('PCA{}.npy'.format(dd), pca.components_)
    return pca, x_list


def evaluate_dd(dd, d=13, k=5, M=8, maxIter=20, epsilon=0, maxS=32):
    ''' The wrapper for training and testing
        Mainly testing dd (i.e., d'), the dimemsion after being transformed by PCA)
        Kept the other hyperparameters to be tunable just in case
    '''
    print("Experimenting with PCA-reduced dimension of {}\nOther parameters: M={}, maxIter={}, epsilon={}, maxS={}".format(
        dd, M, maxIter, epsilon, maxS))

    start = time.time()

    trainThetas = []
    testMFCCs = []
    pca, X_list = fitPCA(d, dd)  # Fit the PCA

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs[0:maxS]:
            # print("Training with speaker {}...".format(speaker))

            files = fnmatch.filter(os.listdir(
                os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = X_list[speaker] # Get the original MFCCs for this speaker
            XX = pca.transform(X) # Use the fitted PCA model to get the input with reduced dimension
            trainThetas.append(train(speaker, XX, M, epsilon, maxIter))

    numCorrect = 0
    fout = open("gmmLiks.txt", 'w').close()  # Clean up the output file
    for i in range(0, len(testMFCCs)):
        XX = pca.transform(testMFCCs[i]) # Also transform the testing data
        numCorrect += test(XX, i, trainThetas, k)

    accuracy = 1.0 * numCorrect / len(testMFCCs)

    end = time.time()
    timeElapsed = end - start

    print("---\nTime:{:4f}s\nAccuracy: {:2f}%\n----------\n".format(timeElapsed, accuracy * 100.))
    return accuracy, timeElapsed


if __name__ == "__main__":

    random.seed(1)

    dd_list = [1, 2, 3, 4, 6, 8, 10]  # up to d = 13

    fout = open('gmmPCA.txt', 'w')

    fout.write('At M = 8, maxIter = 20, epsilon = 0, maxS = 32:\n')
    for dd in dd_list:
        accuracy, timeElapsed = evaluate_dd(dd=dd)
        fout.write("\td' = {}\t=>\taccuracy = {:.4f}, time = {:.4f}s\n".format(
            dd, accuracy, timeElapsed))

    fout.close()
