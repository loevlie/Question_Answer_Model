from Utils import get_features
import binaryAnswers
import QAfeatures
from preprocess import preprocess
import sys
import spacy

from model import ruleBasedModel

import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_md')


context_file = sys.argv[1]
question_file = sys.argv[2]

with open(context_file) as f:
    raw_text = f.read() 

with open(question_file) as g:
    questions = g.read().split('\n')

raw_text = preprocess(raw_text)
fullText = nlp(raw_text)

# f = open('an',"w+")

for q in questions:
    QS = QAfeatures.QuestionSense(q)
    if QS.yes_no:
        if not QS.subject:
            print(QS.doc)
        ans = 'Yes' if binaryAnswers.runThroughSentences(QS,fullText) == True else 'No'
    else:
        featureVectors = get_features(fullText,QS,3)
        ans = ruleBasedModel(featureVectors)
    print(ans)

# f.close()







