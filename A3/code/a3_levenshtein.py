import os
import numpy as np

try:
    dataDir = '/u/cs401/A3/data/'
except:
    dataDir = 'data/'

def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    0.333 0 0 1
    >>> wer("who is there".split(), "".split())
    1.0 0 0 3
    >>> wer("".split(), "who is there".split())
    Inf 0 3 0
    """

    '''Reference: https://web.stanford.edu/~jurafsky/slp3/slides/2_EditDistance.pdf'''
    '''Initialize variables, the table, and define constants'''
    n = len(r)  # reference length
    m = len(h)  # hypothesis length

    R = np.zeros((n+1, m+1))  # forward table
    R[0, :] = np.arange(R[0, :].shape[0])
    R[:, 0] = np.arange(R[:, 0].shape[0])

    B = np.zeros((n+1, m+1))  # backtrace table
    UP, LEFT, UPLEFT = 1, 2, 3
    B[0, :] = LEFT
    B[:, 0] = UP
    B[0, 0] = 0

    '''Forward update to fill the table'''
    for i in range(1, n+1):
        for j in range(1, m+1):
            delete = R[i-1, j] + 1
            substitute = R[i-1, j-1] + (0 if r[i-1] == h[j-1] else 1)
            insert = R[i, j-1] + 1

            R[i, j] = min([delete, substitute, insert])

            if (R[i, j] == delete):
                B[i, j] = UP
            elif (R[i, j] == insert):
                B[i, j] = LEFT
            else:
                B[i, j] = UPLEFT

    '''Backtrace to count the word errors'''
    i, j = n, m
    num_deletes = num_inserts = num_substitutes = 0
    while(i >= 0 and j >= 0 and B[i, j] != 0):
        if (B[i, j] == UP):
            num_deletes += 1
            i -= 1
        elif (B[i, j] == LEFT):
            num_inserts += 1
            j -= 1
        elif (B[i, j] == UPLEFT):
            num_substitutes += (0 if r[i-1] == h[j-1] else 1)
            i -= 1
            j -= 1
        else:
            print("Unexpected condition faced!", B[i, j])

    wer = float(num_substitutes + num_inserts + num_deletes)/float(n) if n != 0 else float("inf")

    return wer, num_substitutes, num_inserts, num_deletes


if __name__ == "__main__":
    HUMAN = "transcripts.txt"
    GOOGLE = "transcripts.Google.txt"
    KALDI = "transcripts.Kaldi.txt"

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            
            # Get the transcript
            humanTranscript = open(os.path.join(dataDir, speaker, HUMAN), "r")
            googleTranscript = open(os.path.join(dataDir, speaker, GOOGLE), "r")
            kaldiTranscript = open(os.path.join(dataDir, speaker, KALDI), "r")

            for kaldiHypo, googleHypo, Ref in zip(kaldiTranscript, googleTranscript, humanTranscript):
                kaldiHypo, googleHypo, Ref = preprocess(
                    kaldiHypo), preprocess(googleHypo), preprocess(Ref)

                wer, numSub, numIns, numDel = Levenshtein(Ref, kaldiHypo)
                print("%5s %6s %2d %4.3f S:%4d, I:%4d, D:%4d" %
                      (speaker, "Kaldi", i, wer, numSub, numIns, numDel))
                wer, numSub, numIns, numDel = Levenshtein(Ref, googleHypo)
                print("%5s %6s %2d %4.3f S:%4d, I:%4d, D:%4d" %
                      (speaker, "Google", i, wer, numSub, numIns, numDel))