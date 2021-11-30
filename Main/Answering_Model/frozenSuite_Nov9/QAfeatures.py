import spacy
nlp = spacy.load('en_core_web_md')

questionWords = ['when','why','how','what','which','where','who','whom','whose']
auxWords = ['be','can','could','do','have','may','might','shall','should','will','would']
subjectPOS = ['nsubj','nsubjpass']
nounPOS = ['NOUN','PROPN','PRON']
skipPOS = ['VERB','ADP','AUX','CCONJ','SCONJ','PART','DET','ADV']

class ParseNode:
    def __init__(self,token,dep,category,parent=None):
        self.token = token
        if type(token) == spacy.tokens.Span:
            self.words = [word.text.lower() for word in token]
        else:
            self.words = [token.text.lower()]
        
        self.dep = dep
        self.category = category
        self.children = []
        self.parent = parent

    def displayTree(self):
        msg = self.token.root.dep_ if type(self.token)==spacy.tokens.Span else self.token.dep_
        msg += '-' + self.token.text
        if self.children:
            out = ''
            for child in self.children:
                if child.category == 'noun':
                    out += '[' + child.displayTree() + '] '
                elif child.category == 'modifier':
                    out += '<' + child.displayTree() + '> '
            out = out[:-1]
            if out:
                msg += ' ' + out
        return msg

    def getChain(self):
        chain = []
        travNode = self
        while travNode.parent != None:
            chain.append((travNode.dep,travNode.parent))
            travNode = travNode.parent
        
        return chain

    def POS(self):
        if type(self.token)==spacy.tokens.Token:
            return self.token.pos_
        return self.token.root.pos_
        

def within(token,child):
    # check whether CHILD is already within TOKEN
    if type(token) == spacy.tokens.Token:
        if type(child) == spacy.tokens.Token:
            return (child == token)
        return False
    elif type(token) == spacy.tokens.Span:
        if type(child) == spacy.tokens.Token:
            return (child in token)
        return (child == token)


class QuestionSense:
    def __init__(self,text):
        self.doc = nlp(text)
        span = spacy.tokens.Span(self.doc,0,len(self.doc))
        self.rootToken = span.root

        self.operativeWord = None
        self.secondaryOp = []
    
        if self.doc[0].lemma_ in auxWords:
            self.yes_no = True
            
        else:
            self.yes_no = False
            wordList = self.doc.text.lower().split()
            operativeWords = [q for q in questionWords if q in wordList]
            if len(operativeWords) > 1:
                operativeWords = [w for w in operativeWords if w != 'when']
                if len(operativeWords) > 1:
                    raise Warning('WARNING: more than one operative word found (' + \
                      ', '.join('"' + w + '"' for w in operativeWords) + ').')
        
            if not operativeWords:
                raise Warning('Error: this question could not be resolved.')
                return
            
            self.operativeWord = operativeWords[0]
            location = wordList.index(self.operativeWord)

        self.ansType = None
        if self.operativeWord == 'when':
            self.ansType = 'TIME'
        elif self.operativeWord == 'where':
            self.ansType = 'LOCATION'
        elif self.operativeWord == 'who' or self.operativeWord == 'whom' or self.operativeWord == 'whose':
            self.ansType = 'PERSON'
        elif self.operativeWord == 'how':
            if location < len(wordList)-1:
                nextWord = wordList[location+1]
                if nextWord == 'many':
                    self.ansType = 'AMT_COUNTABLE'
                    self.secondaryOp.append(nextWord)
                elif nextWord == 'much':
                    self.ansType = 'AMT_UNCOUNTABLE'
                    self.secondaryOp.append(nextWord)
                elif nextWord == 'long' or nextWord == 'old':
                    self.ansType = 'TIME'
                    self.secondaryOp.append(nextWord)
                
    
        nounPhrases = list(self.doc.noun_chunks)
        self.phraseDic = {p.root:p for p in nounPhrases}
        self.nodeDic = {}

        #print('Parse tree for sentence: ' + DFS(rootToken,phraseDic,True))
        self.questionChain = []
        self.descriptors = None
        self.questionNode = None
        
        self.treeRoot = self.DFStree(self.rootToken,'root',rootFlag=True)
        #print('Full parse tree: ' + self.treeRoot.displayTree())
        if self.yes_no:
            self.findComparison()
        return

    def DFStree(self,token,category,parent=None,rootFlag=False):
        dep = token.dep_
        if token in self.phraseDic:
            node = ParseNode(self.phraseDic[token],dep,category,parent)
        else:
            node = ParseNode(token,dep,category,parent)

        children = list(token.children)

        opFlag = (self.operativeWord and (self.operativeWord in node.words) or\
                  any(t.text.lower() == self.operativeWord for t in token.subtree))

        strictOpFlag = (self.operativeWord and (self.operativeWord in node.words))
        
        for child in children:
            if child in self.phraseDic or any(p in child.subtree for p in self.phraseDic):
                node.children.append(self.DFStree(child,'noun',node))
                
            else:
                
                if (rootFlag or opFlag) and not within(node.token,child) and child.dep_ != 'punct':
                    node.children.append(self.DFStree(child,'modifier',node))
                else:
                    continue

        if opFlag and (strictOpFlag or node.POS() not in skipPOS or node.dep in subjectPOS):
            self.analyzeNode(node)

        if token in self.phraseDic:
            self.nodeDic[self.phraseDic[token]] = node
        
        return node


    def analyzeNode(self,node):
        if type(node.token) == spacy.tokens.Token:
            node.words = [w.text for w in node.token.subtree]
            token = nlp(''.join(w.text + w.whitespace_ for w in node.token.subtree))
        else:
            token = node.token
        
        indices = [i for i,word in enumerate(node.words) if \
                      word.lower() != self.operativeWord and word.lower() not in self.secondaryOp]
        if indices:
            self.descriptors = token[indices[0]:]
        else:
            self.descriptors = None
        
        self.questionNode = node
        
        chain = []
        travNode = node
        while travNode.parent != None:
            chain.append((travNode.dep,travNode.parent))
            travNode = travNode.parent
        self.questionChain = chain
        return

    def findComparison(self):
        firstLevel = self.treeRoot.children
        candidatePhrases,modifiers = [],[]
        for node in firstLevel:
            if node.category != 'noun':
                modifiers.append(node)
            else:
                if node.POS() in nounPOS:
                    candidatePhrases.append(node)
                else:
                    modifiers.append(node)
            
        if not candidatePhrases:
            print('Could not resolve this question as a binary comparison')
            self.subject,self.predicate = None,None
            return
        
        candidateSubjects = [node for node in candidatePhrases if node.dep in subjectPOS]
        if candidateSubjects:
            subject = candidateSubjects[0]
        else:
            subject = candidatePhrases[0]

        
        if len(candidateSubjects) > 1 and self.rootToken.lemma_ in auxWords:
            predicate = candidateSubjects[1]
        elif len(candidatePhrases) > 1:
            predicate = candidatePhrases[1]
        else:
            for child in firstLevel:
                if child in candidatePhrases:
                    continue
                candidatePhrases.extend([grandchild for grandchild in child.children\
                        if grandchild.category=='noun' and grandchild.POS() in nounPOS])
            if len(candidatePhrases) > 1:
                predicate = candidatePhrases[1]
            elif modifiers:
                predicate = modifiers[0]
            else:
                for child in candidatePhrases:
                    candidatePhrases.extend([grandchild for grandchild in child.children \
                        if grandchild.category=='noun' and grandchild.POS() in nounPOS])
                if len(candidatePhrases) > 1:
                    predicate = candidatePhrases[1]
                else:
                    predicate = None

        self.gullets = [m for m in modifiers if predicate != m \
                        and predicate.parent != m and m.token.lemma_ not in auxWords]

        gulletText = self.rootToken.text
        
        if predicate and predicate.parent.dep == 'agent':
            subject,predicate = predicate,subject #passive voice!
        elif predicate:
            chainFlipped = predicate.getChain()[::-1]
            if len(chainFlipped) > 1:
                for dep,node in chainFlipped[1:]:
                    if dep != 'appos':
                        gulletText += ' ' + node.token.text
                
            
        self.subject,self.predicate = subject,predicate
        self.gulletText = gulletText
        return

class AnswerSense:
    
    def __init__(self,answer,candidates):
        self.doc = answer
        self.rootToken = self.doc[:].root
        self.candidates = candidates
        # candidates should be a list of root tokens
    
        nounPhrases = list(self.doc.noun_chunks)
        self.phraseDic = {p.root:p for p in nounPhrases}
        self.entDic = {e.root:e for e in self.doc.ents}
        self.nodeDic = {}

        #print('Parse tree for sentence: ' + DFS(rootToken,phraseDic,True))
        self.treeRoot = self.DFStree(self.rootToken,'root')
        #print('Full parse tree: ' + self.treeRoot.displayTree())
        return

    def DFStree(self,token,category,parent=None):
        dep = token.dep_

        if token in self.entDic:
            node = ParseNode(self.entDic[token],dep,category,parent)
        elif token in self.phraseDic:
            node = ParseNode(self.phraseDic[token],dep,category,parent)
        else:
            node = ParseNode(token,dep,category,parent)
        
        children = list(token.children)
        for child in children:
            if child in self.phraseDic or any(p in child.subtree for p in self.phraseDic):
                node.children.append(self.DFStree(child,'noun',node))
                
            elif not within(node.token,child) and child.dep_ != 'punct':
                node.children.append(self.DFStree(child,'modifier',node))
            else:
                continue

        if token in self.candidates:
            # We have a candidate answer
            if token in self.entDic:
                token = self.entDic[token]
            elif token in self.phraseDic:
                token = self.phraseDic[token]
            self.nodeDic[token] = (node,node.getChain())
        
        return node

    def findComparison(self):
        firstLevel = self.treeRoot.children
        candidatePhrases,modifiers = [],[]
        for node in firstLevel:
            if node.category != 'noun':
                modifiers.append(node)
            else:
                if node.POS() in nounPOS:
                    candidatePhrases.append(node)
                else:
                    modifiers.append(node)
            
        if not candidatePhrases:
            #print('Could not resolve this answer as a binary comparison')
            self.subject,self.predicate = None,None
            return
        
        candidateSubjects = [node for node in candidatePhrases if node.dep in subjectPOS]
        if candidateSubjects:
            subject = candidateSubjects[0]
        else:
            subject = candidatePhrases[0]

        
        if len(candidateSubjects) > 1 and self.rootToken.lemma_ in auxWords:
            predicate = candidateSubjects[1]
        elif len(candidatePhrases) > 1:
            predicate = candidatePhrases[1]
        else:
            for child in firstLevel:
                if child in candidatePhrases:
                    continue
                candidatePhrases.extend([grandchild for grandchild in child.children\
                        if grandchild.category=='noun' and grandchild.POS() in nounPOS])
            if len(candidatePhrases) > 1:
                predicate = candidatePhrases[1]
            elif modifiers:
                predicate = modifiers[0]
            else:
                for child in candidatePhrases:
                    candidatePhrases.extend([grandchild for grandchild in child.children \
                        if grandchild.category=='noun' and grandchild.POS() in nounPOS])
                if len(candidatePhrases) > 1:
                    predicate = candidatePhrases[1]
                else:
                    predicate = None

        self.gullets = [m for m in modifiers if predicate != m \
                        and predicate.parent != m and m.token.lemma_ not in auxWords]

        gulletText = self.rootToken.text
        if predicate and predicate.parent.dep == 'agent':
            subject,predicate = predicate,subject #passive voice!
        elif predicate:
            chainFlipped = predicate.getChain()[::-1]
            if len(chainFlipped) > 1:
                for dep,node in chainFlipped[1:]:
                    gulletText += ' ' + node.token.text
            
        self.subject,self.predicate = subject,predicate
        self.gulletText = gulletText
        
        return

        
def verbParent(chain):
    verbs,secondaries = [],[]
    for (dep,node) in chain:
        if type(node.token) != spacy.tokens.Token:
            continue
        secondaries.append(node)
        if node.token.pos_ == 'VERB' or node.token.pos_ == 'AUX':
            verbs.append(node.token)
    nonAuxVerbs = [v for v in verbs if v.lemma_ not in auxWords]
    if nonAuxVerbs:
        return nonAuxVerbs[0]
    for chainNode in secondaries: # if nothing found
        verbs.extend([child.token for child in chainNode.children if \
                      type(child.token)==spacy.tokens.Token and \
                      (child.token.pos_ == 'VERB' or child.token.pos_ =='AUX')])
        # look at all the children along the chain
        
    nonAuxVerbs = [v for v in verbs if v.lemma_ not in auxWords]
    if nonAuxVerbs: # aaand try again
        return nonAuxVerbs[0]
    if verbs: # well, I guess we have only auxiliary verbs
        return verbs[0] # return the closest one
    return None # give up

def fullContext(token):
    lefts,rights = list(token.lefts),list(token.rights)
    if not lefts and not rights:
        return token.text
    msg = ''
    for leftChild in lefts:
        msg += fullContext(leftChild) + ' '
    msg += token.text + ' '
    for rightChild in rights:
        msg += fullContext(rightChild) + ' '
    if msg[-1] == ' ':
        msg = msg[:-1]
    return msg


if __name__ == '__main__':
    import helpers
    while True:
        text = input('Enter a question. >')
        if not text:
            break
        QS = QuestionSense(text)
        print('Root word of sentence: {}'.format(QS.rootToken))
        print('This is {}a yes-no question'.format('NOT ' * int(QS.yes_no == False)))
        if QS.operativeWord:
            print('This is a specific question with operative word "{}"'.format(QS.operativeWord))
            if QS.ansType:
                print('This operative word is usually associated with: {}'.format(QS.ansType))
            else:
                print('This operative word\'s meaning depends on context')
            print('The question is found with direct context: {}'.format(QS.descriptors if QS.descriptors else '[N/A]'))
            print('Downwards dependence of question particle: ' + QS.questionNode.displayTree())
            print('Upwards dependence of question particle: ' + ', which is '.join('{} of word "{}"'.format(t[0],t[1].token) for t in QS.questionChain))
        print('Full parse tree: ' + QS.treeRoot.displayTree())
        if QS.yes_no:
            print('The subject of this binary comparison is: {}'.format(QS.subject.displayTree() if QS.subject else '[N/A]'))
            print('The predicate of this binary comparison is: {}'.format(QS.predicate.displayTree() if QS.predicate else '[N/A]'))
            if QS.subject and QS.predicate:
                print('Central comparison: ' + helpers.displayStructure(QS))
                
        
