ominousWords = ['do','say','think','feel','believe','mean']

def unanswerable(QS):
    if QS.operativeWord == 'why':
        return True
    if QS.operativeWord == 'how' and not QS.ansType:
        return True
    if QS.operativeWord == 'what':
        
        whatToken = [t for t in QS.doc if t.text.lower()=='what'][0]
        if any(t.lemma_ == 'happen' for t in QS.doc[whatToken.i:]):
            
            return True
        if QS.doc[whatToken.i + 1].lemma_ == 'do':
            nextWords = [t.lemma_ for t in QS.doc[whatToken.i + 2:]]
            if any(k in nextWords for k in ominousWords):
                return True
    return False
