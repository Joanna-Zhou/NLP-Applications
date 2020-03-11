import os
import numpy as np
import sys

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

def fetch_transcript(dataDir, speaker, filename):
    """Helper function, just to load the transcripts into a list of lines"""
    filepath = os.path.join(dataDir, speaker, 'transcripts.txt')
    return open(filepath).readlines()


def preproc(string):
    """Preprocess the input sentence by removing punctuations and turn into lists"""
    import re
    string = re.sub(r"[^a-zA-Z0-9\s\[\]]", r"", string)
    string = string.lower()
    string = string.strip()
    string = string.split()

    return string[2:]

if __name__ == "__main__":
    '''Process each line in each transcript file'''
    kaldi_wer = google_wer = output = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print("Processing {}...".format(speaker))

            transcript = fetch_transcript(dataDir, speaker, 'transcripts.txt')
            google_transcript = fetch_transcript(dataDir, speaker, 'transcripts.Google.txt')
            kaldi_transcript = fetch_transcript(dataDir, speaker, 'transcripts.Kaldi.txt')

            for i, ref in enumerate(transcript):
                ref = preproc(ref)

                kaldi = preproc(kaldi_transcript[i])
                kaldi = Levenshtein(ref, kaldi)
                kaldi_wer.append(kaldi[0])
                out.append("[%s] [%s] [%d] [%f] S:[%d] I:[%d] D:[%d]\n" % \
                           (speaker, "Kaldi", i, kaldi[0], kaldi[1], kaldi[2], kaldi[3]))

                google = preproc(google_transcript[i])
                google = Levenshtein(ref, google)
                google_wer.append(google[0])
                out.append("[%s] [%s] [%d] [%f] S:[%d] I:[%d] D:[%d]\n" %
                           (speaker, "Google", i, google[0], google[1], google[2], google[3]))

    '''Log main info'''
    fout = open("asrDiscussion.txt", 'w')
    for line in output:
        # sys.stdout.write(text)
        fout.write(line)

    '''Log statistical info to second last line'''
    fout.write("Kaldi - mean: %f, standard deviation: %f. Google - mean: %f standard deviation: %f." %
               (np.mean(kaldi_wer), np.std(kaldi_wer), np.mean(google_wer), np.std(google_wer)))
    fout.close()
