def unanswerable(QS):
    if QS.operativeWord == 'why':
        return True
    if QS.operativeWord == 'how' and not QS.ansType:
        return True
    return False
