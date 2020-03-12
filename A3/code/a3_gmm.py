# from sklearn.model_selection import train_test_split
import numpy as np
import os
import fnmatch
import random

# added libraries
from scipy.special import logsumexp

# dataDir = '/u/cs401/A3/data/'
dataDir = '/Users/joanna.zyz/NLP-Applications/A3/data'
random.seed(1)

class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


# * Helper functions *
def set_up(myTheta, X):
    ''' Randomize the initial parameters
    '''
    M, d = myTheta.mu.shape
    T = X.shape[0]

    myTheta.omega[:, 0] = 1. / M
    myTheta.mu = X[np.random.choice(T, M)]
    myTheta.Sigma[:, :] = 1.
    return myTheta, M, d, T


def get_preComputedForM(myTheta):
    ''' Precompute all the terms inside log_b_m_x that are independent of x
        Returns an array of length M, each of which can be directly added to the x-dependent part of log_b_m_x
    '''
    M, d = myTheta.mu.shape

    # Numerator terms
    term_mu = - np.sum(np.power(myTheta.mu, 2)/myTheta.Sigma, axis=1)

    # Denominator terms
    term_pi = d * np.log(2.0 * np.pi)
    term_sig = np.sum(np.log(myTheta.Sigma), axis=1)

    preComputedForM = 0.5 * (term_mu - term_pi - term_sig)
    return preComputedForM


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    M, d = myTheta.mu.shape
    axis = 0 if len(x.shape) == 1 else 1
    mu, sig = myTheta.mu[m], myTheta.Sigma[m]

    # Compute the term log_b_m_x depending on if preComputedForM[m] is passed in
    if len(preComputedForM) == 0:
        log_b_m_x = -0.5 * (np.sum(np.square(x - mu)/sig, axis=axis) + d * np.log(2. * np.pi)
                            + np.sum(np.log(sig)))
    else:
        log_b_m_x = -0.5 * (np.sum(x * (x - 2.0 * mu)/sig,
                                   axis=axis)) + preComputedForM[m]

    return log_b_m_x


def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M, d = myTheta.mu.shape

    # Numerator (array/list of M elements)
    weights = myTheta.omega.squeeze()
    preComputedForM = get_preComputedForM(myTheta)
    log_bs = [log_b_m_x(m, x, myTheta, preComputedForM)
              for m in range(M)]

    # Denominator (scalar, the normalization factor)
    log_sum_weighted_log_bs = logsumexp(a=np.array(bs), b=weights)

    return np.log(weights[m]) + log_bs[m] - log_sum_weighted_log_bs


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    weights = myTheta.omega
    # print("weights: {}, log_Bs: {}".format(weights.shape, log_Bs.shape))
    log_P_x_s = logsumexp(a=log_Bs, axis=0, b=weights)  # compute per [m]

    return np.sum(log_P_x_s)  # sum along [t]


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # Setup theta class
    T, d = X.shape
    myTheta = theta(speaker, M, d)
    myTheta, M, d, T = set_up(myTheta, X)

    # Defining constants
    i = 0
    prev_Loss = float('-inf')
    improvement = float('inf')

    while (i <= maxIter and improvement >= epsilon):
        preComputedForM = get_preComputedForM(myTheta)

        '''Compute Intermediate Results'''
        # Get log_b_m_xs in (M, T) for each [m]
        log_b_m_xs = np.zeros((M, T))
        for m in range(M):
            log_b_m_xs[m] = log_b_m_x(m, X, myTheta, preComputedForM)

        # Get log_p_m_xs in (M, T) in vectorized form, also for each [m]
        log_weighted_bs = log_b_m_xs + np.log(myTheta.omega)
        log_normalization = logsumexp(log_weighted_bs, axis=0)
        log_p_m_xs = log_weighted_bs - log_normalization

        '''Compute Likelihood of X from current Theta'''
        loss = logLik(log_b_m_xs, myTheta)

        '''Update parameters'''
        for m in range(M):
            p_m_x = np.exp(log_p_m_xs[m])  # (T,)
            p_m_x_sum = np.sum(p_m_x)  # Scalar (summed over all [t])

            myTheta.omega[m] = p_m_x_sum / T  # Scalar
            myTheta.mu[m] = np.dot(p_m_x, X) / p_m_x_sum  # (d,)
            myTheta.Sigma[m] = np.dot(p_m_x, np.square(
                X)) / p_m_x_sum - np.square(myTheta.mu[m])  # (d,)

        '''Set up for the next iteration'''
        improvement = loss - prev_Loss
        prev_Loss = loss
        i += 1
        # if (i%5 == 1): # checkpoint
        #     print("Iter {}\t| loss = {:.3f}".format(i, prev_Loss))

    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
            [ACTUAL_ID]
            [SNAME1] [LOGLIK1]
            [SNAME2] [LOGLIK2]
            ...
            [SNAMEK] [LOGLIKK]

        e.g.,
            S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    bestLogLik = float('-inf')
    models_and_LogLiks = []
    output = [models[correctID].name+'\n']

    M = models[0].omega.shape[0]
    T, d = mfcc.shape

    '''Find the log likelihoods of the observation (mfcc) given each model (theta)'''
    for i in range(len(models)):
        myTheta = models[i]
        preComputedForM = get_preComputedForM(myTheta)
        log_b_m_xs = np.zeros((M, T))
        for m in range(M):
            log_b_m_xs[m] = log_b_m_x(
                m, mfcc, myTheta, preComputedForM)
        score = logLik(log_b_m_xs, myTheta)
        models_and_LogLiks.append((i, logLik(log_b_m_xs, myTheta)))

    '''Find the best (largest) k likelihoods and their corresponding IDs'''
    sorted_models_and_LogLiks = sorted(models_and_LogLiks, key=lambda x: x[1], reverse=True)
    for i, score in sorted_models_and_LogLiks[0:k]:
        output.append("%s %f\n" % (models[i].name, score))

    '''Log to file'''
    fout = open("gmmLiks.txt", 'a+')
    for line in output:
        fout.write(line)
    fout.write('\n')
    fout.close()

    '''Compare to ground truth'''
    bestModel = sorted_models_and_LogLiks[0][0]
    return 1 if (bestModel == correctID) else 0


def evaluate_hyperparam(d=13, k=5, M=8, maxIter=20, epsilon=0, maxS=32):
    ''' Just a wrapper for the traning and testing code provided in the original code in main
        Returns the test accuracy with the model trained with the current hyper-parameters
    '''
    print("Experimenting with M={}, maxIter={}, epsilon={}, maxS={}...".format(
        M, maxIter, epsilon, maxS))
    trainThetas = []
    testMFCCs = []
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs[0:maxS]:
            # print("Training with speaker {}...".format(speaker))

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    numCorrect = 0
    fout = open("gmmLiks.txt", 'w').close() # Clean up the output file
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)

    accuracy = 1.0 * numCorrect / len(testMFCCs)
    return accuracy

if __name__ == "__main__":

    experimental_Mode = False  # TODO: set to false before submission

    if experimental_Mode:
        # Parameters to be tuned
        M_list = [1, 2, 4, 6, 8]
        maxIter_list = [1, 4, 12, 20]
        epsilon_list = [1e3, 1e4, 1e5]
        maxS_list = [4, 8, 16, 32] # 32 is the default, so skipped

        fout = open('gmmAccuracies.txt', 'w')

        '''Different M'''
        fout.write('At maxIter = 20, epsilon = 0, maxS = 32:\n')
        for M in M_list:
            accuracy = evaluate_hyperparam(M=M, maxIter=10)
            fout.write('\tM = {}\t=>\taccuracy = {:.4f}\n'.format(M, accuracy))

        '''Different maxIter'''
        print('\n-------\n')
        fout.write('\n-------\n')
        fout.write('At M = 4, epsilon = 0, maxS = 32:\n')
        for maxIter in maxIter_list:
            accuracy = evaluate_hyperparam(M=4, maxIter=maxIter)
            fout.write('\tmaxIter = {}\t=>\taccuracy = {:.4f}\n'.format(maxIter, accuracy))

        '''Different epsilon'''
        print('\n-------\n')
        fout.write('\n-------\n')
        fout.write('At M = 4, maxIter = 10, maxS = 32:\n')
        for epsilon in epsilon_list:
            accuracy = evaluate_hyperparam(M=4, maxIter=10, epsilon=epsilon)
            fout.write('\tepsilon = {}\t=>\taccuracy = {:.4f}\n'.format(
                epsilon, accuracy))

        '''Different maxS'''
        print('\n-------\n')
        fout.write('\n-------\n')
        fout.write('At M = 4, maxIter = 10, epsilon = 0:\n')
        for maxS in maxS_list:
            accuracy = evaluate_hyperparam(M=4, maxIter=10, maxS=maxS)
            fout.write('\tmaxS = {}\t=>\taccuracy = {:.4f}\n'.format(
                maxS, accuracy))
        fout.close()

    else:
        M, maxIter, epsilon, maxS = 8, 20, 0, 32
        accuracy = evaluate_hyperparam(M=M, maxIter=maxIter, epsilon=epsilon, maxS=maxS)
        print("Accuracy = {:.4f}\n".format(accuracy))
