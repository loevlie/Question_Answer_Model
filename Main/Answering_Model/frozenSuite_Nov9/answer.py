from Utils import get_features
import binaryAnswers
import QAfeatures
from preprocess import preprocess
import sys
import spacy 

nlp = spacy.load('en_core_web_md')


context_file = sys.argv[1]
question_file = sys.argv[2]

with open(context_file) as f:
    raw_text = f.read() 

with open(question_file) as g:
    questions = g.read().split()

rawText = preprocess(raw_text)
fullText = nlp(rawText)

# f = open('an',"w+")

for q in questions:
    QS = QAfeatures.QuestionSense(q)
    if QS.yes_no:
        ans = binaryAnswers(QS,fullText)
    else:
        featureVectors = get_features(QS,fullText,3)
        # ans = model(featureVectors)
        ans = 'TODO'
    print(ans)

# f.close()








