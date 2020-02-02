# Original dependencies
import sys
import argparse
import os
import json
import re
import spacy

# Added dependencies
import html  # Python3
import string
import time

# Reference: https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                    "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def replace_contractions(text):
    contractions_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    def replace(match):
        return contraction_dict[match.group(0)]
    return contractions_re.sub(replace, text)


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
    # full_preproc = True
    # print_current(comment, 0, 'original')

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
        modComm = re.sub(r"(http|www)\S+", "", modComm)
        # modComm = re.sub(
        #     r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?', '', modComm, flags=re.MULTILINE)
        # modComm = re.sub(r'http\S+', '', modComm, flags=re.MULTILINE)
        # modComm = re.sub(r'www\S+', '', modComm, flags=re.MULTILINE)
        # for replacement in ((' // ', ' '), ('()', ' '), ('[]', ' '), ('( )', ' '), ('[ ]', ' ')):
        #     modComm = modComm.replace(*replacement)
        # print_current(modComm, 3)

    if 4 in steps:  # remove duplicate spaces
        # Each token must now be separated by a single space.
        modComm = re.sub('\s{2,}', ' ', modComm)
        # print_current(modComm, 4)

    # TODO: get Spacy document for modComm
    modComm = replace_contractions(modComm)
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

        # Further refine the text by separating punctuations from text
        doc = re.compile("[\S]+").findall(sent.text)
        senttext = ''
        for token in doc:
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
                # Keep it if it is all cap
                # if len(re.compile("(^| )[A-Z]{3,}").findall(comment)) > 0:
                #     senttext += token.text + '/' + token.tag_
                # Keep it if it doesn't starts with - but its lemma does
                if token.lemma_.startswith('-') and not token.text.startswith('-'):
                    senttext += token.text + '/' + token.tag_
                # Otherwise replace it with its lemma
                else:
                    senttext += token.lemma_ + '/' + token.tag_
            senttext += ' '
        modComm += senttext + '\n'
        # print(senttext + '\n')

    # print_current(modComm, 5, 'final')
    return modComm


def main(args):
    allOutput = []

    # Debug settings
    debug = False
    debug_with_debug_text = False
    debug_with_sample = False

    if debug_with_debug_text:
        debug_text = "What kind of pie? Now I'm hungry!\\nI'm going to admit, as much as I love the /r/women subreddit, I want to indulge in the less serious things about being girly.\\nI agree with the rest of the group; I've got like 3 or 4 bras that I wear all the time, but I have like 10...some are more pretty than functional.\\n\\nAlso, I just want to say that I'm pretty excited about this subreddit!\\nHonestly, I dab a tiny bit of [MAC foundation stick](http://www.maccosmetics.com/product/spp.tmpl?CATEGORY_ID=CAT158&amp;PRODUCT_ID=646) and blend it in.  It works just fine for me.  \\n\\nI have used UD's product, and I did really like it, but I ran out ages ago and now just use MAC.\\nHehe, I second this. I adore Clueless."
        print(preproc1(debug_text))
        return
    if debug_with_sample:
        data = json.load(open("sample_in.json"))

        # TODO: read those lines with something like `j = json.loads(line)`
        for line in data:
            j = json.loads(line)

            # TODO: choose to retain fields from those lines that are relevant to you
            # 'author', 'controversiality', 'score'
            relevant_fields = ['id', 'body']
            j = ({key: j[key] for key in relevant_fields})

            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            modComm = preproc1(j['body'])
        return
    # global abbrev
    # abbrev = ['Ala.', 'Ariz.', 'Assn.', 'Atty.', 'Aug.', 'Ave.', 'Bldg.', 'Blvd.', 'Calif.', 'Capt.', 'Cf.', 'Ch.', 'Co.', 'Col.', 'Colo.', 'Conn.', 'Corp.', 'DR.', 'Dec.', 'Dept.', 'Dist.', 'Dr.', 'Drs.', 'Ed.', 'Eq.', 'FEB.', 'Feb.', 'Fig.', 'Figs.', 'Fla.', 'Ga.', 'Gen.', 'Gov.', 'HON.', 'Ill.', 'Inc.', 'JR.', 'Jan.', 'Jr.', 'Kan.', 'Ky.', 'La.', 'Lt.', 'Ltd.', 'MR.', 'MRS.', 'Mar.', 'Mass.', 'Md.', 'Messrs.', 'Mich.', 'Minn.', 'Miss.', 'Mmes.', 'Mo.', 'Mr.', 'Mrs.', 'Ms.', 'Mx.', 'Mt.', 'NO.', 'No.', 'Nov.', 'Oct.', 'Okla.', 'Op.', 'Ore.', 'Pa.', 'Pp.', 'Prof.', 'Prop.', 'Rd.', 'Ref.', 'Rep.', 'Reps.', 'Rev.', 'Rte.', 'Sen.', 'Sept.', 'Sr.', 'St.', 'Stat.', 'Supt.', 'Tech.', 'Tex.', 'Va.', 'Vol.', 'Wash.', 'al.', 'av.', 'ave.', 'ca.', 'cc.', 'chap.', 'cm.', 'cu.', 'dia.', 'dr.', 'eqn.', 'etc.', 'fig.', 'figs.', 'ft.', 'gm.', 'hr.', 'in.', 'kc.', 'lb.', 'lbs.', 'mg.', 'ml.', 'mm.', 'mv.', 'nw.', 'oz.', 'pl.', 'pp.', 'sec.', 'sq.', 'st.', 'vs.', 'yr.', '', 'e.g.', 'i.e.', 'eg.', 'ie.']
    # abbrev_path = os.path.join(args.a1_dir, 'Wordlists/abbrev.english')
    # abbrev = open(abbrev_path, 'r')
    # abbrev = abbrev.read().split('\n')
    # print(abbrev)

    for subdir, dirs, files in os.walk(indir):
        print("Processing files...")
        for file in files:
            if debug:
                if file != 'Left':
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
                relevant_fields = ['id', 'body']  # 'author', 'controversiality', 'score'
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
        "--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument(
        "--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')

    args = parser.parse_args()

    if (args.max > 200272):
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    # python3 a1_preproc.py 1003002396 -o preproc.json --a1_dir /Users/joanna.zyz/CSC401Assignments/CSC401A1/
    # python3 a1_preproc.py 1003002396 -o preproc_sample.json --a1_dir /Users/joanna.zyz/CSC401Assignments/CSC401A1/ --max 1
    indir = os.path.join(args.a1_dir, 'data')

    global nlp
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)

    main(args)
