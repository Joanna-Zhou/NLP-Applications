# Original dependencies
import numpy as np
import argparse
import json

# Added dependencies
import os
import sys
import re
import string
import time
import csv

# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

# Added lists/dictionaries
CAT_TO_INT = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # Initialize array and separate the tokens vs. tags
    # if debug:
        #  print('\nComment:\n'+comment)
    feats = np.zeros((173,))
    tokens = re.compile(
        "([\w]+|[\W]+)/(?=[\w]+|[\W]+)").findall(comment)

    # TODO: Extract features that rely on capitalization.
    # 1. Number of words in uppercase( 3 letters long)
    uppers = re.compile("(^| )[A-Z]{3,}").findall(comment)
    feats[0] = len(uppers)

    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    comment_lc = comment.lower()
    # 2. Number of first-person pronouns
    # Used re.compile as it is faster than re.match
    # (ref: https://stackoverflow.com/questions/452104/is-it-worth-using-pythons-re-compile)
    fpps = re.compile(
        r'\b(' + r'|'.join(FIRST_PERSON_PRONOUNS) + r')\b').findall(comment_lc)
    feats[1] = len(fpps)

    # 3. Number of second-person pronouns
    spps = re.compile(
        r'\b(' + r'|'.join(SECOND_PERSON_PRONOUNS) + r')\b').findall(comment_lc)
    feats[2] = len(spps)

    # 4. Number of third-person pronouns
    tpps = re.compile(
        r'\b(' + r'|'.join(THIRD_PERSON_PRONOUNS) + r')\b').findall(comment_lc)
    feats[3] = len(tpps)

    # 5. Number of coordinating conjunctions
    ccs = re.compile(r"\/CC( |$)").findall(comment)
    feats[4] = len(ccs)

    # 6. Number of past-tense verbs
    pts = re.compile(r"\/VBD( |$)").findall(comment)
    feats[5] = len(pts)

    # 7. Number of future-tense verbs
    fts = re.compile(r"(^| )(will|'ll|gonna)\/").findall(comment_lc)
    fts_tagged = re.compile(
        r"(^| )going\/\S+ to\/\S+ \S+\/VB( |$)").findall(comment)
    feats[6] = len(fts) + len(fts_tagged)

    # 8. Number of commas
    commas = re.compile(r"(^| ),\/,( |$)").findall(comment_lc)
    feats[7] = len(commas)

    # 9. Number of multi-character punctuation tokens
    mcps = re.compile(
        r"(^| )(([^\s\w]{2,})(\")|([^\s\w]{2,}))\/").findall(comment_lc)
    feats[8] = len(mcps)

    # 10. Number of common nouns
    cnns = re.compile(r"\/(NN|NNS)( |$)").findall(comment)
    feats[9] = len(cnns)

    # 11. Number of proper nouns
    pnns = re.compile(r"\/(NNP|NNPS)( |$)").findall(comment)
    feats[10] = len(pnns)

    # 12. Number of adverbs
    advs = re.compile(r"\/(RB|RBR|RBS)( |$)").findall(comment)
    feats[11] = len(advs)

    # 13. Number of wh - words
    whs = re.compile(r"\/(WDT|WP|WP\$|WRB)( |$)").findall(comment)
    feats[12] = len(whs)

    # 14. Number of slang acronyms
    slangs = re.compile(r'\b(' + r'|'.join(SLANG) + r')\b').findall(comment_lc)
    feats[13] = len(slangs)

    # 15. Average length of sentences, in tokens
    # TODO: decide if i want to keep that extra \n at the end of comment,
    #       if not, +1 in sents and modify that in a1_preproc
    sents = comment_lc.count('\n')
    feats[14] = 0 if (sents == 0) else len(tokens)/sents

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    words = re.compile(r"[^\s\w]*\w\S*\/").findall(comment_lc)
    if len(words) == 0:  # only punctuations, no words
        feats[15] = 0
    else:
        total_len = 0
        for word in words:
            total_len += len(word)
        feats[15] = total_len/len(words) - 1  # Minus the slash after each word

    # 17. Number of sentences.
    feats[16] = sents

    # 18. Average of AoA(100-700) from Bristol, Gilhooly, and Logie norms
    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
    # 21. Standard deviation of AoA(100-700) from Bristol, Gilhooly, and Logie norms
    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    AoA, IMG, FAM = [], [], []
    for token in tokens:
        if token in BGL.keys():
            AoA.append(BGL[token]["AoA"])
            IMG.append(BGL[token]["IMG"])
            FAM.append(BGL[token]["FAM"])
    if len(AoA) > 0:
        feats[17:20] = np.mean(AoA), np.mean(IMG), np.mean(FAM)
        feats[20:23] = np.std(AoA), np.std(IMG), np.std(FAM)

    # 24. Average of V.Mean.Sum from Warringer norms
    # 25. Average of A.Mean.Sum from Warringer norms
    # 26. Average of D.Mean.Sum from Warringer norms
    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    V, A, D = [], [], []
    for token in tokens:
        if token in WARRINGER.keys():
            V.append(WARRINGER[token]["V"])
            A.append(WARRINGER[token]["A"])
            D.append(WARRINGER[token]["D"])
    if len(V) > 0:
        feats[23:26] = np.mean(V), np.mean(D), np.mean(A)
        feats[26:29] = np.std(V), np.std(D), np.std(A)

    # Done extracting the first 1-29 features
    return feats


def extract2(feats, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feats: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    '''
    feats[29:] = LIWC[comment_class][comment_id]
    return feats


def extract(data):
    """Extracts all data

    Arguments:
        data {[type]} -- [description]

    Returns:
        feats {1 x (173+1) ndarray} -- 1 row of feartures
    """
    comment, cat, id = data['body'], data['cat'], data['id']

    feats = np.zeros((173+1,))
    feats[:-1] = extract1(comment)
    feats[:-1] = extract2(feats[:-1], cat, id)
    feats[-1] = CAT_TO_INT[cat]
    return feats


def setup(dir):
    """Set up the global variables (dictionaries)

    Arguments:
        dir {String} -- Path to A1 directory
    """
    print("\nSeting up global dictionaries...")

    global BGL, WARRINGER, LIWC
    BGL, WARRINGER, LIWC = {}, {}, {}

    # Create the dictionaries for the norms first
    try:
        bgl_file = open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', 'r')
        warringer_file = open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', 'r')
    except:
        subdir_wordlists = os.path.join(args.a1_dir, 'wordlists/')
        bgl_file = open(subdir_wordlists +
                        'BristolNorms+GilhoolyLogie.csv', 'r')
        warringer_file = open(subdir_wordlists + 'Ratings_Warriner_et_al.csv', 'r')

    for row in csv.DictReader(bgl_file):
        try:
            BGL[row["WORD"]] = {
                "AoA": float(row["AoA (100-700)"]),
                "IMG": float(row["IMG"]),
                "FAM": float(row["FAM"])
            }
        except:
            pass

    for row in csv.DictReader(warringer_file):
        try:
            WARRINGER[row["Word"]] = {
                "V": float(row["V.Mean.Sum"]),
                "A": float(row["A.Mean.Sum"]),
                "D": float(row["D.Mean.Sum"])
            }
        except:
            pass

    # Now, create the dictionaries for LIWC
    subdir_feats = os.path.join(args.a1_dir, 'feats/')
    for cat in ["Alt", "Center", "Right", "Left"]:
        LIWC[cat] = {}
        id_file = os.path.join(subdir_feats, cat + '_IDs.txt')
        ids = open(id_file)
        data_file = os.path.join(subdir_feats, cat + "_feats.dat.npy")
        data = np.load(data_file)
        for id, row in zip(ids, data):
            LIWC[cat][id.strip()] = row

    print("Finish loading global dictionaries.")

    return


def sanity_check(feats):
    """Check and visualize the output numpy array

    Arguments:
        feats { len(data) x 174 }
    """
    print("+++++++++++++++++++++++++ Sanity Check +++++++++++++++++++++++++")
    print("Shape:", feats.shape)
    print("Mean values on the features:")
    means = np.mean(feats, axis=0)
    print(means)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


def main(args):
    data = json.load(open(args.input))
    data_length = len(data)
    feats = np.zeros((data_length, 173+1))
    checkpoint = 100
    
    global debug
    debug = True

    setup(args.a1_dir)

    print("\nProcessing data...")
    for i in range(data_length):
        feats[i] = extract(data[i])
        if i % checkpoint == 0:
            print("Processing the {}th data out of {}.".format(
                i+1, data_length))
            # if debug:
            #     print(feats[i])

    # TODO: check on the big corpus to make sure there's no column that is all zero!
    if debug:
        sanity_check(feats)
    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument(
        "-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument(
        "-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument(
        "-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)

# python3 a1_extractFeatures.py -i preproc_small.json -o feats_small.npz --a1_dir /Users/joanna.zyz/CSC401Assignments/CSC401A1/
# python3 a1_extractFeatures.py -i preproc_medium.json -o feats_medium.npz --a1_dir /Users/joanna.zyz/CSC401Assignments/CSC401A1/
# python3 a1_extractFeatures.py -i preproc.json -o feats.npz --a1_dir /Users/joanna.zyz/CSC401Assignments/CSC401A1/

# python3 a1_extractFeatures.py -i preproc1.json -o feats1.npz --a1_dir /Users/joanna.zyz/CSC401Assignments/CSC401A1/
