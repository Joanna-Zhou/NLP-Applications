# Original dependencies
import sys
import argparse
import os
import json

# Added dependencies
import re
import html # Python3
# import HTMLParser  # Python2
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


def print_current(comment, step, note=''):
    """
    Prints the current comments for debuging
    """
    if note != '':
        note = ' ('+ note +')'
    print("Step {}{}: \n{}\n".format(step, note, comment))


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    # modComm = ''
    full_preproc = True # Other preprocessing steps like punctuation splitting, etc.
    print_current(comment, 0)

    if 1 in steps:
        # Replace all newline characters with spaces
        comment = comment.replace('\n', '')
        # print_current(comment, 1)

    if 2 in steps:
        # Replace HTML character codes with their ASCII equivalent
        comment = html.unescape(comment) # Python3
        # comment = HTMLParser.HTMLParser().unescape(comment)  # Python2
        # print_current(comment, 2)

    if 3 in steps:
        # Remove all URLs (i.e., tokens beginning with http or www or are in the form or xxx.xxx.xxx).
        # (Reference: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python)
        comment = re.sub(
            r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'http\S+', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'www\S+', '', comment, flags=re.MULTILINE)

        # Remove the remainders from the links
        for replacement in ((' // ', ' '), ('()', ' '), ('[]', ' '), ('( )', ' '), ('[ ]', ' ')):
            comment = comment.replace(*replacement)
        # print_current(comment, 3)

    if 4 in steps:
        # Remove duplicate spaces between tokens.
        # Each token must now be separated by a single space.
        comment = re.sub('\s{2,}', ' ', comment)
        # print_current(comment, 4)

    if full_preproc:
        doc = re.compile("[\S]+").findall(comment)
        ignore_list = abbrev + ['e.g.', 'i.e.']
        comment = ''
        for token in doc:
            if not token in ignore_list:
                token = re.sub(
                    r"[\W]+|[\w']+|[!\"#$%&\(\)*+,-./:;<=>?@[\]^_`{|}~\\]+", lambda pat: pat.group(0)+' ', token)
            comment += token
        # print_current(comment, 4.5)

    # Note that 5, 6, 7 are performed with spaCy
    if 5 in steps:
        # Tagging: Tag each token with its part-of-speech.
        # A tagged token consists of a word, the `/' symbol, and the tag(e.g., dog/NN).
    
        # doc = nlp(comment)
        # comment = ''
        # for token in doc:
        #     # print(token.text, token.tag_, '\n')
        #     comment += str(token.text) + '/' + token.tag_ + ' '
        # print_current(comment, 5.1)
            
        # Method 2: useing regex s.t. grouped punctuations aren't separately tagged
        doc = re.compile("[\S]+").findall(comment)
        doc = spacy.tokens.Doc(nlp.vocab, words=doc)
        doc = nlp.tagger(doc)
        comment = ''
        for token in doc:
            comment += str(token.text) + '/' + token.tag_ + ' '
        print_current(comment, 5)

    if 6 in steps:
        # Lemmatization: Replace the token itself with the “token.lemma”.
        # E.g., words/NNS becomes word/NNS.
        # If the lemma begins with a dash (`-') when the token doesn't (e.g., -PRON- for I）, just keep the token.
        doc = comment.split()
        comment = ''
        for tokenWithTag in doc:
            token = tokenWithTag.split('/')[0]
            if token != '':
                tokenInfo = nlp(token)[0]
                if tokenInfo.lemma_.startswith('-') and not token.startswith('-'):
                    comment += token + '/' + tokenInfo.tag_ + ' '
                else:
                    comment += tokenInfo.lemma_ + '/' + tokenInfo.tag_ + ' '
            else:
                comment += tokenWithTag + ' '
        print_current(comment, 6)
        
        # doc = re.compile("([\w]+|[\W]+)/(?:[\w]+|[\W]+)").findall(comment)
        # doc = spacy.tokens.Doc(nlp.vocab, words=doc)
        # doc = nlp.tagger(doc)
        # for i in range(doc.__len__()):
        #     if str(doc[i]) in comment and doc[i].lemma_[0] != '-':
        #         comment = re.sub(
        #             re.escape(str(doc[i])), doc[i].lemma_, comment)
        # print_current(comment, 6)

    if 7 in steps:
        print('TODO')
    if 8 in steps:
        print('TODO')
    if 9 in steps:
        print('TODO')
    if 10 in steps:
        print('TODO')

    modComm = comment
    return modComm



def main( args ):
    allOutput = []

    # Debug settings
    debug = True # flag for debugging
    debug_with_debug_text = True
    debug_text = "THIS IS  WHY      ESPN    IS  DYING. \n\nhttp: // www.foxnews.com/entertainment/2017/02/15/espn-sued-for-wrongful-termination-by-announcer-after-venus-williams-match-call.html \nSOCIAL JUSTICE, PC CULTURE, AND POLITICS ALL FUCK OFF FROM MY SPORTS!!!\n\n\"When all else fails, go on their subreddit &amp; downvote all of the comments to show them our feelings &amp; hide the truth!\" I'm not even really convinced he lied to Pence, versus was asked to lie to the public to downplay the Russia propaganda.  But as the facts now don't align with the official story, someone had to fall on their sword.  OR... there is something more going on here behind the scenes.   At face value, this seems like something they could have weathered. \n\nOh it takes that long for EO to be made/reviewed?\n\nThat is the narrative on on / reee/politburo \n\[\"Because it's *obviously* just an alt right smear campaign or something...\"\](https: // imgur.com/HgrT8Qm)"
    firstfile = True

    if debug_with_debug_text:
        preproc1(debug_text)
        return

    for subdir, dirs, files in os.walk(indir):
        for file in files:
            # For now, only process one file
            if debug and not firstfile:
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
                j = json.loads(line)
                if debug:
                    print(j['body'])

                # TODO: choose to retain fields from those lines that are relevant to you
                relevant_fields = ['body', 'controversiality', 'author', 'score']  # 'id'
                j = ({key: j[key] for key in relevant_fields})

                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                j['cat'] = file

                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                pre_text = preproc1(j['body'])

                # TODO: replace the 'body' field with the processed text
                # j['body'] = pre_text

                # TODO: append the result to 'allOutput'
                # allOutput.append(j)

            # Finish processing file
            end_time = time.time()
            print("Finish processing {}, time taken: {} seconds.\n\n".format(fullFile, end_time-start_time))

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    # Terminal command: python a1_preproc.py 1003002396 - o preproc.json
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=1)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "You are reading {} lines.".format(args.max))
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
