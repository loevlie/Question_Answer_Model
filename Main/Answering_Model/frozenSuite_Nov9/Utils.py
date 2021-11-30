import spacy
import numpy as np
import QAfeatures,dennyCode_modified,helpers
# from preprocess import preprocess

nlp = spacy.load('en_core_web_md')

def get_features(text,QS,num_rel_sentences):

    #entMapping = {'AMT_COUNTABLE': {'CARDINAL': 0.8218884120171673, 'DATE': 0.03648068669527897, 'EVENT': 0.002145922746781116, 'FAC': 0.002145922746781116, 'GPE': 0.004291845493562232, 'LANGUAGE': 0.002145922746781116, 'LAW': 0.002145922746781116, 'LOC': 0.002145922746781116, 'MONEY': 0.006437768240343348, 'NORP': 0.002145922746781116, 'ORDINAL': 0.002145922746781116, 'ORG': 0.002145922746781116, 'PERCENT': 0.05793991416309013, 'PERSON': 0.002145922746781116, 'PRODUCT': 0.002145922746781116, 'QUANTITY': 0.04291845493562232, 'TIME': 0.006437768240343348, 'WORK_OF_ART': 0.002145922746781116}, 'AMT_UNCOUNTABLE': {'CARDINAL': 0.034482758620689655, 'DATE': 0.02586206896551724, 'EVENT': 0.008620689655172414, 'FAC': 0.008620689655172414, 'GPE': 0.008620689655172414, 'LANGUAGE': 0.008620689655172414, 'LAW': 0.008620689655172414, 'LOC': 0.008620689655172414, 'MONEY': 0.45689655172413796, 'NORP': 0.008620689655172414, 'ORDINAL': 0.008620689655172414, 'ORG': 0.008620689655172414, 'PERCENT': 0.13793103448275862, 'PERSON': 0.008620689655172414, 'PRODUCT': 0.008620689655172414, 'QUANTITY': 0.21551724137931033, 'TIME': 0.02586206896551724, 'WORK_OF_ART': 0.008620689655172414}, 'LOCATION': {'CARDINAL': 0.004807692307692308, 'DATE': 0.008413461538461538, 'EVENT': 0.001201923076923077, 'FAC': 0.057692307692307696, 'GPE': 0.5336538461538461, 'LANGUAGE': 0.001201923076923077, 'LAW': 0.002403846153846154, 'LOC': 0.04567307692307692, 'MONEY': 0.001201923076923077, 'NORP': 0.01201923076923077, 'ORDINAL': 0.027644230769230768, 'ORG': 0.19951923076923078, 'PERCENT': 0.001201923076923077, 'PERSON': 0.09134615384615384, 'PRODUCT': 0.004807692307692308, 'QUANTITY': 0.003605769230769231, 'TIME': 0.002403846153846154, 'WORK_OF_ART': 0.001201923076923077}, 'PERSON': {'CARDINAL': 0.0005777007510109763, 'DATE': 0.002310803004043905, 'EVENT': 0.00028885037550548814, 'FAC': 0.0008665511265164644, 'GPE': 0.046216060080878106, 'LANGUAGE': 0.00028885037550548814, 'LAW': 0.0005777007510109763, 'LOC': 0.0005777007510109763, 'MONEY': 0.00028885037550548814, 'NORP': 0.019641825534373193, 'ORDINAL': 0.00028885037550548814, 'ORG': 0.1582900057770075, 'PERCENT': 0.00028885037550548814, 'PERSON': 0.7651646447140381, 'PRODUCT': 0.0034662045060658577, 'QUANTITY': 0.00028885037550548814, 'TIME': 0.00028885037550548814, 'WORK_OF_ART': 0.00028885037550548814}, 'TIME': {'CARDINAL': 0.10420475319926874, 'DATE': 0.8692870201096892, 'EVENT': 0.0018281535648994515, 'FAC': 0.0009140767824497258, 'GPE': 0.003656307129798903, 'LANGUAGE': 0.0004570383912248629, 'LAW': 0.0009140767824497258, 'LOC': 0.0004570383912248629, 'MONEY': 0.0018281535648994515, 'NORP': 0.0004570383912248629, 'ORDINAL': 0.0009140767824497258, 'ORG': 0.005941499085923218, 'PERCENT': 0.0004570383912248629, 'PERSON': 0.004113345521023766, 'PRODUCT': 0.0018281535648994515, 'QUANTITY': 0.0004570383912248629, 'TIME': 0.0018281535648994515, 'WORK_OF_ART': 0.0004570383912248629}}
    entMapping = {'TIME':['DATE','CARDINAL','TIME'],
                  'LOCATION':['GPE','LOC','FAC','ORG'],
                  'PERSON':['PERSON','ORG','NORP'],
                  'AMT_COUNTABLE':['QUANTITY','MONEY','CARDINAL','PERCENT'],
                  'AMT_UNCOUNTABLE':['QUANTITY','MONEY','CARDINAL','PERCENT']}
    
    skipList = ['INTJ','PUNCT','AUX','ADP','DET','PRON','CCONJ','SCONJ','PART']

    shortExplanations = {'CARDINAL':'numeral',
                     'DATE':'date','EVENT':'event','FAC':'building, road',
                     'GPE':'country, state, city','LANGUAGE':'language',
                     'LAW':'law','LOC':'location','MONEY':'money','NORP':'politics, nation',
                     'ORDINAL':'first, second','ORG':'organization','PERCENT':'percentage',
                     'PERSON':'person','PRODUCT':'product','QUANTITY':'quantity','TIME':'time',
                     'WORK_OF_ART':'artwork'}

#     if text.endswith('.txt'):
#         Answer_File = text # 'messi.txt'
#         with open(Answer_File,'r') as f:
#             rawText = f.read()
#     else:
#         rawText = text
    
#     rawText = rawText.replace('\n','.') # Replace with "rawText = preprocess(rawText)"
    
    sentenceDict = dennyCode_modified.find_similar_sentences(text,QS.doc,num_rel_sentences) # num_rel_sentences = 3 --> This is basically a hyper-parameter
    #print('\n'.join((sentenceDict[i].text.strip()+ ' -- score ' + str(i)) for i in sentenceDict))        

    if QS.yes_no:
        return []
    
    Q_verbParent = QAfeatures.verbParent(QS.questionChain)

    print('\n')
    vectorDict = {}
    
    for i,score in enumerate(sentenceDict):
        sentence = sentenceDict[score]
        print('Candidate sentence: {} ({})'.format(sentence,score))
        if QS.ansType:
            v0 = 1
            candidates = [ent.root for ent in sentence.ents if ent.root.pos_ != 'PRON']
        elif QS.descriptors:
            v0 = 2
            candidates = [p.root for p in sentence.noun_chunks if p.root.pos_ != 'PRON']
        else:
            v0 = 3
            candidates = [p.root for p in sentence.noun_chunks if p.root.pos_ != 'PRON']
##            for token in sentence:
##                if token.pos_ in skipList:
##                    if token in candidates:
##                        candidates.remove(token)
##                    continue
##                if token not in candidates and not any(token in p for p in sentence.noun_chunks):
##                    candidates.append(token)

        AS = QAfeatures.AnswerSense(sentence,candidates)
        #print('Answers from this sentence: ' + '; '.join(i.text for i in AS.nodeDic))
        

        for candidate in AS.nodeDic:

            # Fill out the feature vector, [v0 v1 v2 v3 v4 v5 v6 v7]
            # v0: type of question (easy = 1, medium = 2, hard = 3)
            # v1: similarity between descriptor and candidate (default 0)
            # v2: similarity between candidate's verb parent and question's verb parent
            # v3: fraction of downwards dependents of question particle that are shared
            #     by the candidate (default 1)
            # v4: whether the candidate answer is contained within the question
            # v5: the length of the candidate answer's chain to its root
            # v6: if the question has an answer type, whether the candidate named-entity
            #     is of the type needed (default 0)
            # v7: similarity score of answer sentence
            # v8: average positional distance between candidate and the named entities
            #     in answer sentence (used for tiebreaks only)
            
            node,chain = AS.nodeDic[candidate]

            if QS.descriptors:
                if candidate in AS.doc.ents:
                    alternate = nlp(shortExplanations[candidate.label_])
                    #v1 = max(candidate.similarity(QS.descriptors),\
                    #         alternate.similarity(QS.descriptors))
                    v1 = max(helpers.fixedSimilarity(QS.descriptors,candidate),\
                             helpers.fixedSimilarity(QS.descriptors,alternate))
                else:
                    #v1 = candidate.similarity(QS.descriptors)
                    v1 = helpers.fixedSimilarity(QS.descriptors,candidate)
            else:
                v1 = 0

            A_verbParent = QAfeatures.verbParent(chain)
            if not Q_verbParent or not A_verbParent:
                v2 = 0
            elif Q_verbParent.lemma_ in QAfeatures.auxWords:
                v2 = 0.5
            else:
                v2 = A_verbParent.similarity(Q_verbParent)
                if v2 < 0.5:
                    otherVerbSim = [(t.similarity(Q_verbParent) if \
                                    (t.lemma_ != Q_verbParent.lemma_) else 1) \
                                    for t in AS.doc if t.pos_ == 'VERB']
                    
                    if otherVerbSim and max(otherVerbSim) > 0.8:
                        v2 = 0.8 * max(otherVerbSim)

                if v2 < 0.3:
                    otherQverbs = [(A_verbParent.similarity(t) if \
                                   (t.lemma_ != A_verbParent.lemma_) else 1) \
                                   for t in QS.doc if t.pos_ == 'VERB']
                    
                    if otherQverbs and max(otherQverbs) > 0.8:
                        v2 = 0.6 * max(otherQverbs)

            if not QS.questionNode.children:
                v3 = 1
            else:
                q = set()
                for parseNode in QS.questionNode.children:
                    token = parseNode.token.root if type(parseNode.token)==spacy.tokens.Span \
                            else parseNode.token
                    q.add(token.text)

                r = set()
                for parseNode in node.children:
                    token = parseNode.token.root if type(parseNode.token)==spacy.tokens.Span \
                            else parseNode.token
                    r.add(token.text)

                v3 = len(q & r)/len(q)

            fixedAnswer = '~|'.join(c.lemma_ for c in candidate) \
                      if type(candidate) == spacy.tokens.Span else candidate.lemma_
            fixedQ = '~|'.join(t.lemma_ for t in QS.doc)
            v4 = int(fixedAnswer in fixedQ)

            if v4 == 0 and fixedAnswer.lower() in fixedQ.lower():
                v4 = 0.8
            
            if v4 == 0 and type(candidate)==spacy.tokens.Span and len(candidate) > 1:
                shortFixedLeft = '~|'.join(c.lemma_ for c in candidate[:-1])
                shortFixedRight = '~|'.join(c.lemma_ for c in candidate[1:])
                
                if shortFixedLeft in fixedQ or shortFixedRight in fixedQ:
                    v4 = (len(candidate) - 1.)/(len(candidate))

            v5 = len(chain)

            v6 = 0
            if QS.ansType and candidate in AS.doc.ents and candidate.label_ in entMapping[QS.ansType]:
                v6 = 1

            v7 = score

            overlapEnts = [ent for ent in AS.doc.ents if any(qEnt.text == ent.text for qEnt in QS.doc.ents)]
            v8 = 0
            for ent in overlapEnts:
                distance = abs(ent[0].i - candidate[0].i)
                v8 += distance

            v9 = 0
            ourChildren = node.children
            ourSiblings = node.parent.children if node.parent else []
            if any(c.dep == 'neg' for c in ourChildren) or any(sib.dep == 'neg' for sib in ourSiblings):
                v9 = 1

            vec = np.array([v0,v1,v2,v3,v4,v5,v6,v7,v8,v9])
            #print(candidate,vec)
            vectorDict[candidate] = vec
        #print('Vectors for sentence {}:'.format(i+1), vectors)
    return vectorDict
