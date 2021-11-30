#!/usr/bin/env python3
if __name__ == '__main__':
    from Utils import get_features
    import binaryAnswers
    import QAfeatures
    #from preprocess import preprocess
    import sys
    import spacy
    import csv

    from model import neuralNetModel

    import warnings
    warnings.filterwarnings('ignore')

    nlp = spacy.load('en_core_web_md')


    #context_file = sys.argv[1]
    #question_file = sys.argv[2]

    context_file = 'lincolnCoref.txt'
    question_file = 'questionTest.txt'
    
    with open(context_file,encoding='ISO-8859-1') as f:
        raw_text = f.read()

    with open(question_file) as g:
        questions = g.readlines()
##        reader = csv.reader(g)
##        questions = []
##        for num,q,actualAnswer in reader:
##            questions.append((num,q,actualAnswer))

    #raw_text = preprocess(raw_text)
    fullText = nlp(raw_text)

    # f = open('an',"w+")

    for q in questions:
        print('\nQUESTION: {}'.format(q))
        QS = QAfeatures.QuestionSense(q)
        if QS.yes_no:
            print('Shunt to binary Q/A')
            if not QS.subject:
                print('Failed to parse this question: {}'.format(q))
                continue
            ans = 'Yes' if binaryAnswers.runThroughSentences(QS,fullText) == True else 'No'
        else:
            print('Shunt to specific Q/A')
            if not QS.questionNode:
                print('Failed to parse this question: {}'.format(q))
                continue
            featureVectors = get_features(fullText,QS,3)
##            if not any(key.text == actualAnswer for key in featureVectors):
##                print('WARNING: actual answer not in candidates')
            ans = neuralNetModel(featureVectors)

            if ans == None:
                # we didn't find a good answer; try again with more sentences
                featureVectors = get_features(fullText,QS,6)
##                if not any(key.text == actualAnswer for key in featureVectors):
##                    print('WARNING: actual answer not in candidates')
                ans = ruleBasedModel(featureVectors)

            if ans == None:
                ans = '[NO ANSWER FOUND]'
            else:
                fullAns = QAfeatures.fullContext(ans)
                if not QS.ansType and len(fullAns.split()) < 20:
                    ans = fullAns
                else:
                    ans = ans.text
        print('We think the answer is: ' + ans)
        #print('Actual answer: ' + actualAnswer)







