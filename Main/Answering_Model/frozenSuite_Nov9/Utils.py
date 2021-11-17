import spacy
import numpy as np
from collections import defaultdict
import QAfeatures,dennyCode_modified
# from preprocess import preprocess 

def get_features(text,question,num_rel_sentences):

    entMapping = {'TIME':['DATE','CARDINAL','TIME'],
                  'LOCATION':['GPE','LOC'],
                  'PERSON':['PERSON','ORG'],
                  'AMT_COUNTABLE':['QUANTITY','MONEY','CARDINAL'],
                  'AMT_UNCOUNTABLE':['QUANTITY','MONEY','CARDINAL']}

    if text.endswith('.txt'):
        Answer_File = text # 'messi.txt'
        with open(Answer_File,'r') as f:
            rawText = f.read()
    else:
        rawText = text
    
    rawText = rawText.replace('\n','.') # Replace with "rawText = preprocess(rawText)"

    question = question # 'What disease was Messi diagnosed with?'

    sentenceDict = dennyCode_modified.find_similar_sentences(rawText,question,num_rel_sentences) # num_rel_sentences = 3 --> This is basically a hyper-parameter
    #print('\n'.join((sentenceDict[i].text.strip()+ ' -- score ' + str(i)) for i in sentenceDict))        


    QS = QAfeatures.QuestionSense(question)
    Q_verbParent = QAfeatures.verbParent(QS.questionChain)

    #print('\n')
    vectors = defaultdict(lambda: defaultdict())
    for i,score in enumerate(sentenceDict):
        sentence = sentenceDict[score]
        if QS.ansType:
            candidates = [ent.root for ent in sentence.ents]
        elif QS.descriptors:
            candidates = [p.root for p in sentence.noun_chunks]
        else:
            candidates = [p.root for p in sentence.noun_chunks]
            for token in sentence:
                if token not in candidates and not any(token in p for p in sentence.noun_chunks):
                    candidates.append(token)

        AS = QAfeatures.AnswerSense(sentence,candidates)
        #vectors = {}
        

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
                v1 = candidate.similarity(QS.descriptors)
            else:
                v1 = 0

            A_verbParent = QAfeatures.verbParent(chain)
            if not Q_verbParent or not A_verbParent:
                v2 = 0
            else:
                v2 = A_verbParent.similarity(Q_verbParent)

            if not QS.questionNode.children:
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

            v4 = int('~|'.join(c.text for c in candidate) in \
                     '~|'.join(t.text for t in QS.doc))

            v5 = len(chain)

            v6 = 0
            if QS.ansType and candidate in AS.doc.ents and candidate.label_ in entMapping[QS.ansType]:
                v6 = 1

            v7 = len(candidate)

            vec = np.array([v1,v2,v3,v4,v5,v6,v7])
            #print(candidate,vec)
            vectors['sentence '+str(i)][candidate] = vec
        #print('Vectors for sentence {}:'.format(i+1), vectors['sentence '+str(i)])
    return [vectors['sentence '+str(i)] for i in range(len(vectors))]