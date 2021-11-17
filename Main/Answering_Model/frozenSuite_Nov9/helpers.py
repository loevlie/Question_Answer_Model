import spacy
nlp = spacy.load('en_core_web_md')

joiningPOS = ['PUNCT','CCONJ','SCONJ']

def fixedSimilarity(span1,span2):
    if type(span1) == spacy.tokens.Token or len(span1) <= 1:
        comp1 = span1
    else:
        fixed1 = ' '.join(token.text for token in span1 if not token.is_stop)
        comp1 = nlp(fixed1) if fixed1 else span1

    if type(span2) == spacy.tokens.Token or len(span2) <= 1:
        comp2 = span2
    else:
        fixed2 = ' '.join(token.text for token in span2 if not token.is_stop)
        comp2 = nlp(fixed2) if fixed2 else span2
    
    return comp1.similarity(comp2)


def subTree(word,signposts):
    L = [child for child in word.children if child not in signposts]
    for child in word.children:
        if child in signposts:
            continue
        L.extend(subTree(child,signposts))
    return L

def displayStructure(senseObj):
    if not senseObj.subject or not senseObj.predicate:
        return ''
    msg = senseObj.subject.displayTree()
    msg += ' -- '
    if senseObj.gullets:
        gulletStr = '<' + '> <'.join(gullet.displayTree() for \
                gullet in senseObj.gullets) + '> '
    else:
        gulletStr = ''
    msg += gulletStr + (senseObj.gulletText if senseObj.gulletText else senseObj.rootToken.text)
    msg += ' -> ' + \
           (senseObj.predicate.displayTree() if senseObj.predicate else '[NO PREDICATE]')
    return msg
    
def splitIntoClauses(doc):
    #print('Trying to split "{}"'.format(doc))
    root = doc[:].root
    
    subj = [c for c in root.children if c.dep_ == 'nsubj' or c.dep_ == 'nsubjpass' or c.dep_ == 'attr']
    if not subj:
        return [doc]
    
    primaryVerbs = []
    if root.pos_ == 'VERB' or root.pos_=='AUX':
        primaryVerbs.append(root)
    
    primaryVerbs += [c for c in root.children if c.pos_ == 'VERB' or c.pos_ == 'AUX']
    
    for verb in primaryVerbs:
        if verb == root:
            continue
        dropSpan = doc[min(root.i,verb.i):max(root.i,verb.i)]
        drop = True
        for word in dropSpan:
            if word.pos_ in joiningPOS:
                drop = False
        if drop:
            primaryVerbs.remove(verb)
    if len(primaryVerbs) <= 1:
        return [doc]
    clauses = []
    for verb in primaryVerbs:
        
        startSpan,endSpan = verb.i,verb.i
        for child in subTree(verb,primaryVerbs):
            if child.dep_ == 'punct':
                continue
            if child.i < startSpan:
                startSpan = child.i
            if child.i > endSpan:
                endSpan = child.i
        candidateClause = doc[startSpan:endSpan+1]
        if candidateClause.text.strip():
            clauses.append(candidateClause)
    nClauses = []
    
    for clause in clauses:
        if len(clause) < 3:
            continue
        clauseSubj = [c for c in clause.root.children if c.dep_ == 'nsubj' or c.dep_ == 'nsubjpass' or c.dep_ =='attr']
        if not clauseSubj and clause[0].pos_ not in joiningPOS:
            #print('Want to append a subject to clause "{}"'.format(clause))
            subjToken = subj[0]
            clause = nlp(subjToken.text + ' ' + clause.text)
        nClauses.append(clause)
    return nClauses

def splitClausesFully(doc):
    s = splitIntoClauses(doc)
    while True:
        newS = []
        #print('\n'.join(i.text for i in s) + '\n\n')
        for i in s:
            for j in splitIntoClauses(nlp(i.text)):
                #print(i,'-->',j)
                if not any(k.text == j.text for k in newS):
                    newS.append(j[:])
        if len(newS) <= len(s):
            return s
        s = newS

if __name__ == '__main__':
    while True:
        text = input('Enter a sentence. >')
        if not text:
            break
        doc = nlp(text)
        clauses = splitClausesFully(doc)
        for i in clauses:
            print(i)
