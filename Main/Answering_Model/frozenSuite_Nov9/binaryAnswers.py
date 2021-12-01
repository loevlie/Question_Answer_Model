import spacy
import numpy as np

##with open('lincolnCoref.txt',mode='r',encoding="ISO-8859-1") as f:
##    rawText = f.read()

nlp = spacy.load('en_core_web_md')
##doc = nlp(rawText)

import QAfeatures
import helpers

uselessGullets = ['ever','sometimes','once']

def runThroughSentences(QS,doc):
    subjectToken = QS.subject.token
    subjectHead = subjectToken if type(subjectToken)==spacy.tokens.Token \
                  else subjectToken.root

    if QS.predicate:
        predicateToken = QS.predicate.token
        predicateHead = predicateToken if type(predicateToken)==spacy.tokens.Token \
                        else predicateToken.root

    subjectFixed = nlp(subjectHead.lemma_)
    if QS.predicate:
        predicateFixed = nlp(predicateHead.lemma_)

    sentenceList = []
    k = None
    
    for s in doc.sents:
        subjMatches = [word for word in s if word.lemma_ == subjectHead.lemma_]
        if QS.predicate:
            predMatches = [word for word in s if word.lemma_ == predicateHead.lemma_]
        else:
            predMatches = []
        
        if not subjMatches and not predMatches:
            continue
        elif subjMatches and not predMatches:
            if QS.predicate:
                predVec = np.array([word.similarity(predicateFixed) for word in s if word.text.strip()])
                if np.max(predVec) < 0.85:
                    continue
            
        elif predMatches and not subjMatches:
            subjVec = np.array([word.similarity(subjectFixed) for word in s if word.text.strip()])
            if np.max(subjVec) < 0.85:
                continue
            
        k = compareStructures(QS,(s,False))
        #print('\n')
        
        if k != None:
            break

        #print('Trying to split "{}"'.format(s))
        for clause in helpers.splitClausesFully(nlp(s.text)):
            if clause.text == s.text:
                continue
            k = compareStructures(QS,(clause,True))
            #print('\n')
            if k != None:
                break
        if k != None:
            break
        
    if k != None:
        return k
    #print('No conclusive evidence. We guess NO.')
    return False

def compareStructures(QS,tup):
    sentence,clauseFlag = tup
##    if subjToken and predToken:
##        startSpan,endSpan = min(subjToken.i - sentence[0].i,predToken.i - sentence[0].i),\
##                            max(subjToken.i - sentence[0].i,predToken.i - sentence[0].i)
##
##        childList = list(subjToken.subtree) + list(predToken.subtree) + \
##                    [list(subjToken.ancestors)[0]] + [list(predToken.ancestors)[0]]
##        for child in childList:
##            if child.dep_ == 'punct':
##                continue
##            if child.i - sentence[0].i < startSpan:
##                startSpan = child.i - sentence[0].i
##            if child.i - sentence[0].i > endSpan:
##                endSpan = child.i - sentence[0].i
##        clause = sentence[startSpan:endSpan+1]
##        print('{} --> {}'.format(sentence,clause))
##        clause = nlp(clause.text)

##    if clauseFlag:
##        print('CLAUSE: {}'.format(sentence))
##    else:
##        print('SENTENCE: {}'.format(sentence))

    doc = nlp(sentence.text)
    
    AS = QAfeatures.AnswerSense(doc,[p.root for p in doc.noun_chunks])
    AS.findComparison()
    
##    print('Question: ' + helpers.displayStructure(QS))
##    print('Answer: ' + helpers.displayStructure(AS))
    
    if not AS.subject or (QS.predicate and not AS.predicate):
        #print('Unable to make sense of clause as a binary comparison. Sentence is useless.')
        return None
    
    if AS.rootToken.pos_ != 'VERB' and AS.rootToken.pos_ != 'AUX':
        #print('Unable to find a verb in clause. Sentence is useless.')
        return None

    fixedQsubj,fixedAsubj = QS.subject.token,AS.subject.token
    if type(QS.subject.token) == spacy.tokens.Span:
        if type(AS.subject.token) == spacy.tokens.Token or len(AS.subject.token) <= 1:
            fixedQsubj = QS.subject.token.root
    if type(AS.subject.token) == spacy.tokens.Span:
        if type(QS.subject.token) == spacy.tokens.Token or len(QS.subject.token) <= 1:
            fixedAsubj = AS.subject.token.root
            
    sim = fixedQsubj.similarity(fixedAsubj)
    if sim < 0.8:
        if AS.rootToken.lemma_ == 'be' and AS.predicate.token.similarity(QS.subject.token) >= 0.85:
            #print('Flip subject/predicate of answer')
            AS.subject,AS.predicate = AS.predicate,AS.subject
            fixedAsubj = AS.subject.token
            #print('Answer is now: ' + helpers.displayStructure(AS))
        else:
            #print('Subject mismatch ({} <-> {} = {}). Sentence is useless.'.format(fixedQsubj,fixedAsubj,sim))
            
            return None
    #print('Subjects match ({} <-> {} = {}).'.format(fixedQsubj,fixedAsubj,sim))

    if QS.predicate:
        return matchPredicates(QS,AS)
    else:
        return matchGullets(QS,AS)

def matchPredicates(QS,AS):
    sim = helpers.fixedSimilarity(QS.predicate.token,AS.predicate.token)
    if sim >= 0.85:
        #print('Predicates match ({} <-> {} = {}).'.format(QS.predicate.token,\
        #                                    AS.predicate.token, sim))
        return matchGullets(QS,AS)
    else:
        #print('Predicates do not match  ({} <-> {} = {}). Trying to fix.'.format(QS.predicate.token,\
        #                                   AS.predicate.token,sim))
        usingQmatchA = [p for p in AS.nodeDic if helpers.fixedSimilarity(p,QS.predicate.token) >= 0.85 and p.root.dep_ != 'nsubj']
        usingAmatchQ = [p for p in QS.doc.noun_chunks if helpers.fixedSimilarity(p,AS.predicate.token) >= 0.85]
        if not usingQmatchA and not usingAmatchQ:
            #print('Cannot fix predicate disparity. Sentence is useless.')
            return None
        elif usingAmatchQ:
            # refactor question
            #print('We have found "{}" in question to match answer predicate. We need to refactor the question.'.format(usingAmatchQ[0]))
            match = QS.nodeDic[usingAmatchQ[0]]
            chain = match.getChain()
            nodeChain = [tup[1] for tup in chain]
            
            if QS.predicate in nodeChain:
                #print('Node is a dependent of question predicate.')
                gulletText = QS.rootToken.text
                flip = False
                for node in nodeChain[::-1]:
                    if node == QS.predicate:
                        flip = True
                    if not flip:
                        continue
                    gulletText += ' ' + node.token.text
                QS.predicate = match
                QS.gulletText = gulletText
                #print('Attempted refactor: ' + helpers.displayStructure(QS))

            elif QS.subject == match or QS.subject in nodeChain:
                #print('Node is a dependent of question subject. We don\'t really know what to do here.')
                return None
            
            else:
                #print('Node is in question gullet.')
                try:
                    gulletParent = nodeChain[-2]
                except IndexError:
                    gulletParent = QS.treeRoot
                    
                if gulletParent.POS() == 'CCONJ' or gulletParent.POS() == 'SCONJ':
                    #print('Parent is a conjunction. This should be handled through clauses.')
                    return None
                elif gulletParent.POS() == 'ADP':
                    #print('Parent is a preposition. Add the predicate to the gullet.')
                    gulletText = QS.rootToken.text + ' ' + QS.predicate.token.text
                    for node in nodeChain[-2::-1]:
                        gulletText += ' ' + node.token.text
                    QS.predicate = match
                    if gulletParent in QS.gullets:
                        QS.gullets.remove(gulletParent)

                    QS.gulletText = gulletText
                    #print('Attempted refactor: ' + helpers.displayStructure(QS))
                else:
                    #print('Parent is something else. Boot predicate to gullet and replace.')
                    QS.gullets.append(QS.predicate)
                    if gulletParent in QS.gullets:
                        QS.gullets.remove(gulletParent)

                    gulletText = QS.rootToken.text if gulletParent.POS() != 'VERB' else ''
                    for node in nodeChain[-2::-1]:
                        gulletText += ' ' + node.token.text

                    QS.predicate = match
                    QS.gulletText = gulletText
                    #print('Attempted refactor: ' + helpers.displayStructure(QS))
                
        else:
            # refactor answer
            #print('We have found "{}" in answer to match question predicate. We need to refactor the answer.'.format(usingQmatchA[0]))
            match = AS.nodeDic[usingQmatchA[0]]
            chain = match[1]
            nodeChain = [tup[1] for tup in chain]

            if AS.predicate in nodeChain:
                #print('Node is a dependent of answer predicate.')
                gulletText = AS.rootToken.text
                flip = False
                for node in nodeChain[::-1]:
                    if node == AS.predicate:
                        flip = True
                    if not flip:
                        continue
                    gulletText += ' ' + node.token.text
                AS.predicate = match[0]
                AS.gulletText = gulletText
                #print('Attempted refactor: ' + helpers.displayStructure(AS))

            elif AS.subject == match or AS.subject in nodeChain:
                #print('Node is a dependent of answer subject. We don\'t really know what to do here.')
                return None
            
            else:
                #print('Node is in answer gullet.')
                gulletParent = nodeChain[-2]
                if gulletParent.POS() == 'CCONJ' or gulletParent.POS() == 'SCONJ':
                    #print('Parent is a conjunction. This should be handled through clauses.')
                    return None
                elif gulletParent.POS() == 'ADP':
                    #print('Parent is a preposition. Add the predicate to the gullet.')
                    gulletText = AS.rootToken.text + ' ' + AS.predicate.token.text
                    for node in nodeChain[-2::-1]:
                        gulletText += ' ' + node.token.text
                    AS.predicate = match[0]
                    if gulletParent in AS.gullets:
                        AS.gullets.remove(gulletParent)

                    AS.gulletText = gulletText
                        
                    #print('Attempted refactor: ' + helpers.displayStructure(AS))
                else:
                    #print('Parent is something else. Boot predicate to gullet and replace.')
                    AS.gullets.append(AS.predicate)
                    if gulletParent in AS.gullets:
                        AS.gullets.remove(gulletParent)

                    gulletText = AS.rootToken.text if gulletParent.POS() != 'VERB' else ''
                    for node in nodeChain[-2::-1]:
                        gulletText += ' ' + node.token.text

                    AS.gulletText = gulletText
                    AS.predicate = match[0]
                    #print('Attempted refactor: ' + helpers.displayStructure(AS))

    #print('Predicates now match ({} <-> {} = {})'.format(QS.predicate.token,\
    #        AS.predicate.token, helpers.fixedSimilarity(QS.predicate.token,AS.predicate.token)))
    return matchGullets(QS,AS)

def matchGullets(QS,AS):
    Qgullet,Agullet = nlp(QS.gulletText),nlp(AS.gulletText)
    sim = helpers.fixedSimilarity(Qgullet,Agullet)
    if sim <= 0.5:
        #print('Gullets do not match ({} <-> {} = {}). Sentence is useless.'.format(Qgullet,Agullet,sim))
        return None
    #print('Gullets kind of match ({} <-> {} = {}).'.format(Qgullet,Agullet,sim))
    
    if any(Qnode.dep == 'neg' for Qnode in QS.gullets) and not any(node.dep == 'neg' for node in AS.gullets):
        #print('Question directly negates answer sentence. Answer NO.')
        return False
    elif any(node.dep == 'neg' for node in AS.gullets) and not any(Qnode.dep=='neg' for Qnode in QS.gullets):
        #print('Answer directly negates question sentence. Answer NO.')
        return False
    
    for particleNode in QS.gullets:
        if particleNode.token.text in uselessGullets or particleNode.POS() == 'AUX':
            continue
        elif particleNode.POS() == 'ADP':
            particleNode = particleNode.children[0]
            answerMatches = [p for p in AS.doc.noun_chunks if helpers.fixedSimilarity(p,particleNode.token) >= 0.8]
            if not answerMatches:
                #print('Cannot find question sub-gullet "{}". Sentence is useless.'.format(particleNode.token))
                return None
        else:
            answerMatches = [node for node in AS.gullets if helpers.fixedSimilarity(node.token,particleNode.token) >= 0.8]

            if not answerMatches:
                #print('Cannot find question gullet "{}". Sentence is useless.'.format(particleNode.token))
                return None

    #print('Q/A pair has passed all basic matches of subject, predicate, and gullet. Answer YES.')
    return True
            

if __name__ == '__main__':
    while True:
        question = input('Enter a question. >')
        if not question:
            break
        QS = QAfeatures.QuestionSense(question)
        runThroughSentences(QS,doc)
        
