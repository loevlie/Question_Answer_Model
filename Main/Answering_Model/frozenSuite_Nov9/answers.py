import spacy
import numpy as np
import QAfeatures,dennyCode_modified
# import preprocess
import NN_Model_Use

nlp = spacy.load('en_core_web_md')

entMapping = {'TIME':['DATE','CARDINAL','TIME'],
              'LOCATION':['GPE','LOC','FAC'],
              'PERSON':['PERSON','ORG'],
              'AMT_COUNTABLE':['QUANTITY','MONEY','CARDINAL'],
              'AMT_UNCOUNTABLE':['QUANTITY','MONEY','CARDINAL']}

shortExplanations = {'CARDINAL':'numeral',
                     'DATE':'date','EVENT':'event','FAC':'building, road',
                     'GPE':'country, state, city','LANGUAGE':'language',
                     'LAW':'law','LOC':'location','MONEY':'money','NORP':'politics, nation',
                     'ORDINAL':'first, second','ORG':'organization','PERCENT':'percentage',
                     'PERSON':'person','PRODUCT':'product','QUANTITY':'quantity','TIME':'time',
                     'WORK_OF_ART':'artwork'}


Answer_File = 'lincolnCoref.txt'
with open(Answer_File,'r',encoding='ISO-8859-1') as f:
    rawText = f.read()

#rawText = preprocess.preprocess(rawText)

question = 'Which general was routed by Lee at Chancellorsville?'

sentenceDict = dennyCode_modified.find_similar_sentences(rawText,question,3)
print('\n'.join((sentenceDict[i].text.strip()+ ' -- score ' + str(i)) for i in sentenceDict))        


QS = QAfeatures.QuestionSense(question)
Q_verbParent = QAfeatures.verbParent(QS.questionChain)

vectorList = []
skipList = ['INTJ','PUNCT','AUX','ADP','DET','PRON','CCONJ','SCONJ','PART']

print('\n')
for i,score in enumerate(sentenceDict):
    print('\nSENTENCE {}:'.format(i))
    sentence = sentenceDict[score]
    if QS.ansType:
        candidates = [ent.root for ent in sentence.ents]
    elif QS.descriptors:
        candidates = [p.root for p in sentence.noun_chunks]
    else:
        candidates = [p.root for p in sentence.noun_chunks]
        for token in sentence:
            if token.pos_ in skipList:

                if token in candidates:
                    candidates.remove(token)
                
                continue
            print(token,token.pos_)
            if token not in candidates and not any(token in p for p in sentence.noun_chunks):
                candidates.append(token)
    
    AS = QAfeatures.AnswerSense(sentence,candidates)
    vectors = {}
    
    for candidate in AS.nodeDic:
        
        # Fill out the feature vector, [v1 v2 v3 v4 v5 v6 v7]
        # v1: similarity between descriptor and candidate (default 0)
        # v2: similarity between candidate's verb parent and question's verb parent
        # v3: fraction of downwards dependents of question particle that are shared
        #     by the candidate (default 1)
        # v4: whether the candidate answer is contained within the question
        # v5: the length of the candidate answer's chain to its root
        # v6: if the question has an answer type, whether the candidate named-entity
        #     is of the type needed (default 0)
        # v7: the length of the candidate answer
        node,chain = AS.nodeDic[candidate]
        
        if QS.descriptors:
            if candidate in AS.doc.ents:
                alternate = nlp(shortExplanations[candidate.label_])
                v1 = max(candidate.similarity(QS.descriptors),\
                         alternate.similarity(QS.descriptors))
            else:
                v1 = candidate.similarity(QS.descriptors)
        else:
            v1 = 0

        A_verbParent = QAfeatures.verbParent(chain)
        if not Q_verbParent or not A_verbParent:
            v2 = 0
        else:
            v2 = A_verbParent.similarity(Q_verbParent)

        if not QS.questionNode or not QS.questionNode.children:
            v3 = 1
        else:
            q = set()
            for parseNode in QS.questionNode.children:
                token = parseNode.token.root if type(parseNode.token)==spacy.tokens.Span \
                        else parseNode.token
                q.add(token.text)

            r = set()
            for parseNode in candidate.children:
                token = parseNode.token.root if type(parseNode.token)==spacy.tokens.Span \
                        else parseNode.token
                r.add(token.text)

            v3 = len(q.intersect(r))/len(q)

        fixedAnswer = '~|'.join(c.lemma_ for c in candidate) \
                      if type(candidate) == spacy.tokens.Span else candidate.lemma_
        fixedQ = '~|'.join(t.lemma_ for t in QS.doc)
        v4 = int(fixedAnswer in fixedQ)
        if v4 == 0 and type(candidate)==spacy.tokens.Span and len(candidate) > 1:
            shortFixedLeft = '~|'.join(c.lemma_ for c in candidate[:-1])
            shortFixedRight = '~|'.join(c.lemma_ for c in candidate[1:])
            if shortFixedLeft in fixedQ or shortFixedRight in fixedQ:
                v4 = (len(candidate) - 1.)/(len(candidate))

        v5 = len(chain)

        v6 = 0
        if QS.ansType and candidate in AS.doc.ents and candidate.label_ in entMapping[QS.ansType]:
            v6 = 1

        v7 = len(candidate)

        vec = np.array([v1,v2,v3,v4,v5,v6,v7])
        #print(candidate,vec)
        vectors[candidate] = vec
        print(candidate)
        print(vec)
        print(NN_Model_Use.getProbability(vec))
        
    vectorList.append(vectors)
    

