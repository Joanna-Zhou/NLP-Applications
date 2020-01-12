# Original dependencies
import numpy as np
import argparse
import json

# Added dependencies
import sys
import json

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


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    # Initialize array and separate the tokens vs. tags
    feats = np.zeros((1, 173))
    tokens = re.compile(
        "([\w]+|[\W]+)/(?=[\w]+|[\W]+)").findall(comment)
    tags = re.compile(
        "(?=[\w]+|[\W]+)/([\w]+|[\W]+)").findall(comment)

    # TODO: Extract features that rely on capitalization.
    # 1. Number of words in uppercase( 3 letters long)
    for e in body:
        if e.isupper and len(e)>3:
            feats[0][0] += 1

    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    comment_lc = comment.lower()
    # 2. Number of first-person pronouns
    # Used re.compile as it is faster than re.match
    # (ref: https://stackoverflow.com/questions/452104/is-it-worth-using-pythons-re-compile)
    fpp_list = ['i','me','my','mine','we','us','our','ours']
    fpps = re.compile(r'\b(' + r'|'.join(fpp_list) + r')\b').findall(comment_lc)
    feats[0][1] = len(fpps)
    
    # 3. Number of second-person pronouns
    spp_list = ['you', 'your', 'yours', 'u', 'ur', 'urs']
    spps = re.compile(r'\b(' + r'|'.join(spp_list) + r')\b').findall(comment_lc)
    print(spps)
    feats[0][2] = len(spps)
    
    # 4. Number of third-person pronouns
    tpp_list = ['he', 'him', 'his', 'she', 'her', 'hers',
                'it', 'its', 'they', 'them', 'their', 'theirs']
    tpps = re.compile(r'\b(' + r'|'.join(tpp_list) + r')\b').findall(comment_lc)
    feats[0][3] = len(tpps)

    # 5. Number of coordinating conjunctions
    # 6. Number of past-tense verbs
    # 7. Number of future-tense verbs
    # 8. Number of commas
    # 9. Number of multi-character punctuation tokens
    # 10. Number of common nouns
    # 11. Number of proper nouns
    # 12. Number of adverbs
    # 13. Number of wh - words
    # 14. Number of slang acronyms
    # 15. Average length of sentences, in tokens
    # 16. Average length of tokens, excluding punctuation-only tokens, in characters
    # 17. Number of sentences.
    # 18. Average of AoA(100-700) from Bristol, Gilhooly, and Logie norms
    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
    # 21. Standard deviation of AoA(100-700) from Bristol, Gilhooly, and Logie norms
    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    # 24. Average of V.Mean.Sum from Warringer norms
    # 25. Average of A.Mean.Sum from Warringer norms
    # 26. Average of D.Mean.Sum from Warringer norms
    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    # TODO: Extract features that do not rely on capitalization.
    print('TODO')


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
    print('TODO')


def main(args):
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    # TODO: Use extract1 to find the first 29 features for each
    # data point. Add these to feats.
    # TODO: Use extract2 to copy LIWC features (features 30-173)
    # into feats. (Note that these rely on each data point's class,
    # which is why we can't add them in extract1).
    print('TODO')

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()

    main(args)

