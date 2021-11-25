import spacy
import numpy as np
import QAfeatures,dennyCode_modified,helpers
# from preprocess import preprocess

nlp = spacy.load('en_core_web_md')

def get_features(text,QS,num_rel_sentences):

    entMapping = {'TIME':['DATE','CARDINAL','TIME'],
                  'LOCATION':['GPE','LOC','FAC'],
                  'PERSON':['PERSON','ORG'],
                  'AMT_COUNTABLE':['QUANTITY','MONEY','CARDINAL'],
                  'AMT_UNCOUNTABLE':['QUANTITY','MONEY','CARDINAL']}
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
    rawText = text
    sentenceDict = dennyCode_modified.find_similar_sentences(rawText,QS.doc,num_rel_sentences) # num_rel_sentences = 3 --> This is basically a hyper-parameter
    #print('\n'.join((sentenceDict[i].text.strip()+ ' -- score ' + str(i)) for i in sentenceDict))        

    if QS.yes_no:
        return []
    
    Q_verbParent = QAfeatures.verbParent(QS.questionChain)

    #print('\n')
    vectorDict = {}

    if QS.doc.text.startswith('How long'):
        QS.ansType = 'TIME'
    elif QS.doc.text.startswith('How far'):
        QS.ansType = 'AMT_COUNTABLE'
    
    for i,score in enumerate(sentenceDict):
        sentence = sentenceDict[score]
        #print('Candidate sentence: {} ({})'.format(sentence,score))
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
                    
                    if max(otherVerbSim) > 0.8:
                        v2 = 0.8 * max(otherVerbSim)

                if v2 < 0.3:
                    otherQverbs = [(A_verbParent.similarity(t) if \
                                   (t.lemma_ != A_verbParent.lemma_) else 1) \
                                   for t in QS.doc if t.pos_ == 'VERB']
                    
                    if max(otherQverbs) > 0.8:
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
            ourSiblings = node.parent.children
            if any(c.dep == 'neg' for c in ourChildren) or any(sib.dep == 'neg' for sib in ourSiblings):
                v9 = 1

            vec = np.array([v0,v1,v2,v3,v4,v5,v6,v7,v8,v9])
            #print(candidate,vec)
            vectorDict[candidate] = vec
        #print('Vectors for sentence {}:'.format(i+1), vectors)
    return vectorDict
