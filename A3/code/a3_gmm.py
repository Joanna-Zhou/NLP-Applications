from sklearn.model_selection import train_test_split
import numpy as np
import os
import fnmatch
import random

# added libraries
from scipy.special import logsumexp

# dataDir = '/u/cs401/A3/data/'
dataDir = '/Users/joanna.zyz/NLP-Applications/A3/data'


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

        self.setup(self, M, d)

    def setup(self, M, d, X):
        ''' Set up preComputedForM and randomize the initial parameters
        '''
        self.M = M
        self.d = d
        self.omega[:, 0] = 1. / M
        self.mu = X[np.random.choice(X.shape[0], M)] # (T, M)
        self.Sigma[:, :] = 1.
        # self.preComputedForM = get_preComputedForM(M, d)

    def set_preComputedForM(self):
        ''' Precompute all the terms inside log_b_m_x that are independent of x
            Returns an array of length M, each of which can be directly added to the x-dependent part of log_b_m_x
        '''
        # TODO: check is i am supposed to move this out of the function
        # Numerator terms
        term_mu = - np.sum(np.power(self.mu, 2)/self.Sigma, axis=1)

        # Denominator terms
        term_pi = self.d * np.log(2.0 * np.pi)
        term_sig = np.sum(np.log(self.Sigma), axis=1)

        preComputedForM = 0.5 * (term_mu - term_pi - term_sig)
        self.preComputedForM = preComputedForM

    def get_preComputedForM(self):
        ''' In case I need to use preComputedForM without recalculating it'''
        try:
            return self.preComputedForM
        except:
            self.set_preComputedForM(self, self.M, self.d)
            return self.preComputedForM


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM
    '''
    M, d=myTheta.mu.shape
    axis=0 if len(x.shape) == 1 else 1
    mu, sig=myTheta.mu[m].squeeze(), myTheta.Sigma[m].squeeze()
    print("x: {}, mu: {}, mu before squeeze: {}".format(
        x.shape, mu.shape, myTheta.mu[m].shape))

    # Compute the term log_b_m_x depending on if preComputedForM[m] is passed in
    if len(preComputedForM) == 0:
        log_b_m_x=-0.5 * (np.sum(np.square(x - mu)/sig, axis=axis) + d * np.log(2. * np.pi)
                          + np.sum(np.log(sig)))
    else:
        log_b_m_x=-0.5 * (np.sum(x * (x - 2.0 * mu)/sig, axis=axis)) + preComputedForM[m]

    return log_b_m_x


def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    M, d = myTheta.mu.shape

    # Numerator (array/list of M elements)
    weights = myTheta.omega.squeeze()
    log_bs = [log_b_m_x(m, x, myTheta, myTheta.preComputedForM) for m in range(M)]

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
    print("weights: {}, log_Bs: {}".format(weights.shape, log_Bs.shape))
    log_Ps = logsumexp(a=log_Bs, axis=0, b=weights) # compute per [m]

    return np.sum(log_Ps) # sum along [t]


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    # Setup theta class
    T, d = X.shape
    myTheta = theta(speaker, M, d)
    myTheta.setup(M, d, X)

    # Defining constants
    i = 0
    loss_prev = loss_diff = float('-inf')

    while (i <= maxIter and abs(loss_diff) >= epsilon):
        print("Iter {} | loss = {}".format(i, loss_prev))
        myTheta.set_preComputedForM()
        
        T = X.shape[0]

        log_Bs = np.zeros((M, T))
        log_Ps = np.zeros((M, T))

        for m in range(M):
            log_Bs[m, :] = log_b_m_x(m, X, myTheta, preComputedForM)

        log_Ps_N_r = np.add(np.log(myTheta.omega),  log_Bs)
        log_Ps = np.add(log_Ps_N_r, -logsumexp(log_Ps_N_r, axis=0))

        log_Bs, log_Ps = compute_intermediate_results(
            X, M, myTheta, preComputedForM)
        loss = logLik(log_Bs, myTheta)
        myTheta = update_params(myTheta, X, log_Ps, loss)
        loss_diff = loss - loss_prev
        loss_prev = loss
        i += 1

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
    bestModel=-1
    print('TODO')
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas=[]
    testMFCCs=[]
    print('TODO: you will need to modify this main block for Sec 2.3')
    d=13
    k=5  # number of top speakers to display, <= 0 if none
    M=8
    epsilon=0.0
    maxIter=20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files=fnmatch.filter(os.listdir(
                os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC=np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X=np.empty((0, d))
            for file in files:
                myMFCC=np.load(os.path.join(dataDir, speaker, file))
                X=np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect=0;
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy=1.0*numCorrect/len(testMFCCs)
