#!/usr/bin/env python3
if __name__ == '__main__':
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
                print('Binary question failed parse: {}'.format(q))
                continue
            ans = 'Yes' if binaryAnswers.runThroughSentences(QS,fullText) == True else 'No'
        else:
            if not QS.questionNode:
                print('Specific question failed parse: {}'.format(q))
                continue

            if unanswerable(QS):
                bestDict = dennyCode_modified.find_similar_sentences(fullText,QS.doc,1)
                ans = list(bestDict.values())[0].text

            else:
                
                featureVectors = get_features(fullText,QS,3)
                ans = ruleBasedModel(featureVectors)

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
            
        print(ans)

    # f.close()






