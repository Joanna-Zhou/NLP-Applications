# Original dependencies
import sys
import argparse
import os
import json
import re
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

# Added dependencies
import html  # Python3
import string
import time

#abbrev_path = '/u/cs401/Wordlists/abbrev.english'
abbrev_path = '../wordlists/abbrev.english'
abbrev = open(abbrev_path, 'r')
abbrev = abbrev.read().split('\n')

def print_current(comment, step, note=''):
    """
    Prints the current comments for debuging
    """
    if note != '':
        note = ' (' + note + ')'
    print("Step {}{}: \n{}\n".format(step, note, comment))


def preproc1(comment, steps=range(1, 5)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    # Other preprocessing steps like punctuation splitting, etc.
    full_preproc = 0
    if full_preproc:
        ignore_list = abbrev + ['e.g.', 'i.e.']
    print_current(comment, 0, 'original')

    modComm = comment
    if 1 in steps:  # replace newlines with spaces
        modComm = re.sub(r"\n{1,}", " ", modComm)
        # print_current(modComm, 1)

    if 2 in steps:  # unescape html
        # Replace HTML character codes with their ASCII equivalent
        modComm = html.unescape(modComm)  # Python3
        # print_current(modComm, 2)

    if 3 in steps:  # remove URLs
        # Remove all URLs (i.e., tokens beginning with http or www or are in the form or xxx.xxx.xxx).
        # (Reference: https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python)
        # modComm = re.sub(r"(http|http|www).*\s", "", modComm)
        if full_preproc:
            modComm = re.sub(
                r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', '', modComm, flags=re.MULTILINE)
            modComm = re.sub(r'http\S+', '', modComm, flags=re.MULTILINE)
            modComm = re.sub(r'www\S+', '', modComm, flags=re.MULTILINE)
            for replacement in ((' // ', ' '), ('()', ' '), ('[]', ' '), ('( )', ' '), ('[ ]', ' ')):
                modComm = modComm.replace(*replacement)
        # print_current(modComm, 3)

    if 4 in steps:  # remove duplicate spaces
        # Each token must now be separated by a single space.
        modComm = re.sub('\s{2,}', ' ', modComm)
        # print_current(modComm, 4)

    # TODO: get Spacy document for modComm
    utt = nlp(modComm)
    modComm = ''

    # TODO: use Spacy document for modComm to create a string.
    # Make sure to:
    #    * Insert "\n" between sentences.
    #    * Split tokens with spaces.
    #    * Write "/POS" after each token.
    for sent in utt.sents:
        senttext = sent.text
        if senttext == '': # Empty line
            continue

        if full_preproc:
            # Further refine the text by separating punctuations from text
            doc = re.compile("[\S]+").findall(sent.text)
            senttext = ''
            for token in doc:
                if not token in ignore_list:
                    token = re.sub(
                        r"[\W]+|[\w']+|[!\"#$%&\(\)*+,-./:;<=>?@[\]^_`{|}~\\]+", lambda pat: pat.group(0)+' ', token)
                senttext += token

        # Tagging: Tag each token with its part-of-speech.
        # A tagged token consists of a word, the `/' symbol, and the tag(e.g., dog/NN).
        doc = re.compile("[\S]+").findall(senttext)
        doc = spacy.tokens.Doc(nlp.vocab, words=doc)
        doc = nlp.tagger(doc)
        senttext = ''
        for token in doc:
            if token.text != '':
                if token.lemma_.startswith('-') and not token.text.startswith('-'):
                    senttext += token.text + '/' + token.tag_
                else:
                    senttext += token.lemma_ + '/' + token.tag_
            senttext += ' '
        modComm += senttext + '\n'

    print_current(modComm, 5, 'final')
    return modComm


def main(args):
    allOutput = []

    # Debug settings
    debug = True
    debug_with_debug_text = False
    debug_text = "THIS IS  WHY      ESPN    IS  DYING. \n\nhttp: // www.foxnews.com/entertainment/2017/02/15/espn-sued-for-wrongful-termination-by-announcer-after-venus-williams-match-call.html \nSOCIAL JUSTICE, PC CULTURE, AND POLITICS ALL FUCK OFF FROM MY SPORTS!!!\n\n\"When all else fails, go on their subreddit &amp; downvote all of the comments to show them our feelings &amp; hide the truth!\" I'm not even really convinced he lied to Pence, versus was asked to lie to the public to downplay the Russia propaganda.  But as the facts now don't align with the official story, someone had to fall on their sword.  OR... there is something more going on here behind the scenes.   At face value, this seems like something they could have weathered. \n\nOh it takes that long for EO to be made/reviewed?\n\nThat is the narrative on on / reee/politburo \n\[\"Because it's *obviously* just an alt right smear campaign or something...\"\](https: // imgur.com/HgrT8Qm)"

    if debug_with_debug_text:
        preproc1(debug_text)
        return

    for subdir, dirs, files in os.walk(indir):
        print("Processing files...")
        for file in files:
            if debug and file != 'Left':
                continue

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

                # TODO: choose to retain fields from those lines that are relevant to you
                relevant_fields = [
                    'body', 'controversiality', 'id', 'score']  # 'author'
                j = ({key: j[key] for key in relevant_fields})

                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                j['cat'] = file

                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                modComm = preproc1(j['body'])

                # TODO: replace the 'body' field with the processed text
                j['body'] = modComm

                # TODO: append the result to 'allOutput'
                allOutput.append(j)

            # Finish processing file
            end_time = time.time()
            print("Finish processing {} comments from the {}, time taken: {} seconds.\n\n".format(
                args.max, file, end_time-start_time))

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument(
        "-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument(
        "--max", type=int, help="The maximum number of comments to read from each file", default=1)
    parser.add_argument(
        "--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    # python3 a1_preproc.py 1003002396 -o preproc.json --a1_dir /Users/joanna.zyz/CSC401Assignments/CSC401A1/
    indir = os.path.join(args.a1_dir, 'data')
    # abbrev_path = os.path.join(args.a1_dir, 'wordlists/abbrev.english')
    main(args)
