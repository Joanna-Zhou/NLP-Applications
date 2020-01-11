# Original dependencies
import sys
import argparse
import os
import json

# Added dependencies
import re
import html # Python3
import HTMLParser  # Python2
import string
import time
import spacy
nlp = spacy.load('en', disable = ['parser','ner'])

indir = '/Users/joanna.zyz/CSC401Assignments/CSC401A1/data/';

#abbrev_path = '/u/cs401/Wordlists/abbrev.english'
abbrev_path = '../wordlists/abbrev.english'
abbrev = open(abbrev_path, 'r')
abbrev = abbrev.read().split('\n')

#StopWords_path = '/u/cs401/Wordlists/StopWords'
StopWords_path = '../wordlists/StopWords'
StopWords = open(StopWords_path, 'r')
StopWords = StopWords.read().split('\n')


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    # Finishing flags
    step6, step8 = False, False

    modComm = ''
    if 1 in steps:
        # Replace all newline characters with spaces
        comment = comment.replace('\n', '')
        print("Type:", type(comment))

    if 2 in steps:
        # Replace HTML character codes with their ASCII equivalent
        # comment = html.unescape(comment) # Python3
        comment = HTMLParser.HTMLParser().unescape(comment)  # Python2
        

    if 3 in steps:
        # Remove all URLs (i.e., tokens beginning with http or www or are in the form or xxx.xxx.xxx).
        # (Reference: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python)
        comment = re.sub(
            r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'http\S+', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'www\S+', '', comment, flags=re.MULTILINE)
        # comment = comment.replace('()', ' ')
        # comment = comment.replace('[]', ' ')

    if 4 in steps:
        # Remove duplicate spaces between tokens.
        # Each token must now be separated by a single space.
        pass

    # Note that 5, 6, 7 are performed with spaCy
    if 5 in steps:
        # Tagging: Tag each token with its part-of-speech.
        # A tagged token consists of a word, the `/' symbol, and the tag(e.g., dog/NN).
        print('TODO')
    if 6 in steps:
        print('TODO')
    if 7 in steps:
        print('TODO')
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')

    return modComm

def main( args ):

    allOutput = []
    firstfile = True
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            # For now, only process one file
            if not firstfile: 
                break
            firstfile = False
            
            # Start recording time
            print("Processing {}...".format(file))
            start_time = time.time()

            fullFile = os.path.join(subdir, file)
            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines - lines: a list of strings
            lines = data[args.ID[0] %
                         len(data): args.ID[0] % len(data) + args.max]

            # TODO: read those lines with something like `j = json.loads(line)`
            for line in lines:
                print(line)
                j = json.loads(line)

            # TODO: choose to retain fields from those lines that are relevant to you
            relevant_fields = ['body', 'controversiality', 'author', 'score']  # 'id'
            j = ({key: j[key] for key in relevant_fields})

            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            j['cat'] = file

            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            pre_text = preproc1(j['body'])

            # TODO: replace the 'body' field with the processed text
            j['body'] = pre_text

            # TODO: append the result to 'allOutput'
            # allOutput.append(j)

            # Finish processing file
            end_time = time.time()
            print("Finish processing {}, time taken: {} seconds.\n\n".format(fullFile, end_time-start_time))

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


def print_current(comment, step):
    """Prints the current comments for debuging

    Arguments:
        comment {[type]} -- [description]
        step {[type]} -- [description]
    """
    pass

if __name__ == "__main__":
    # Terminal command: python a1_preproc.py 1003002396 - o preproc.json
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "You are reading {} lines.".format(args.max))
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
