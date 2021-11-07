import spacy
nlp = spacy.load('en_core_web_sm')

questionWords = ['when','why','how','what','which','where','who','whom','whose']
auxWords = ['be','can','could','do','have','may','might','shall','should','will','would']

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
            chain.append((travNode.dep,travNode.parent.token))
            travNode = travNode.parent
        
        return chain
        

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
        elif self.operativeWord == 'why':
            self.ansType = 'REASON'
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
    
        nounPhrases = list(self.doc.noun_chunks)
        self.phraseDic = {p.root:p for p in nounPhrases}
        self.nodeDic = {}

        #print('Parse tree for sentence: ' + DFS(rootToken,phraseDic,True))
        self.treeRoot = self.DFStree(self.rootToken,'root',rootFlag=True)
        #print('Full parse tree: ' + self.treeRoot.displayTree())
        return

    def DFStree(self,token,category,parent=None,rootFlag=False):
        dep = token.dep_
        if token in self.phraseDic:
            node = ParseNode(self.phraseDic[token],dep,category,parent)
        else:
            node = ParseNode(token,dep,category,parent)

        opFlag = (self.operativeWord and (self.operativeWord in node.words))
        children = list(token.children)
        for child in children:
            if child in self.phraseDic or any(p in child.subtree for p in self.phraseDic):
                node.children.append(self.DFStree(child,'noun',node))
                
            else:
                
                if (rootFlag or opFlag) and not within(node.token,child):
                    node.children.append(self.DFStree(child,'modifier',node))
                else:
                    continue

        if opFlag:
            self.analyzeNode(node)

        if token in self.phraseDic:
            self.nodeDic[self.phraseDic[token]] = node
        
        return node


    def analyzeNode(self,node):
        indices = [i for i,word in enumerate(node.words) if \
                      word.lower() != self.operativeWord and word.lower() not in self.secondaryOp]
        if indices:
            self.descriptors = node.token[indices[0]:]
        else:
            self.descriptors = None
        
        self.questionNode = node
        
        chain = []
        travNode = node
        while travNode.parent != None:
            chain.append((travNode.dep,travNode.parent.token))
            travNode = travNode.parent
        self.questionChain = chain
        return

class AnswerSense:
    
    def __init__(self,answer,candidates):
        self.doc = answer
        self.rootToken = self.doc[:].root
        self.candidates = candidates
        # candidates should be a list of root tokens
    
        nounPhrases = list(self.doc.noun_chunks)
        self.phraseDic = {p.root:p for p in nounPhrases}
        self.nodeDic = {}

        #print('Parse tree for sentence: ' + DFS(rootToken,phraseDic,True))
        self.treeRoot = self.DFStree(self.rootToken,'root')
        #print('Full parse tree: ' + self.treeRoot.displayTree())
        return

    def DFStree(self,token,category,parent=None):
        dep = token.dep_
        if token in self.phraseDic:
            node = ParseNode(self.phraseDic[token],dep,category,parent)
        else:
            node = ParseNode(token,dep,category,parent)
        
        children = list(token.children)
        for child in children:
            if child in self.phraseDic or any(p in child.subtree for p in self.phraseDic):
                node.children.append(self.DFStree(child,'noun',node))
                
            else:
                node.children.append(self.DFStree(child,'modifier',node))

        if token in self.candidates:
            # We have a candidate answer
            self.nodeDic[token] = (node,node.getChain())
        
        return node

        
    
if __name__ == '__main__':
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
            print('Upwards dependence of question particle: ' + ', which is'.join('{} of word "{}"'.format(t[0],t[1]) for t in QS.questionChain))
        print('Full parse tree: ' + QS.treeRoot.displayTree())
        
