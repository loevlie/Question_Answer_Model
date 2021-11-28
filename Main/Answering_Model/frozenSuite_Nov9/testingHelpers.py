import spacy
nlp = spacy.load('en_core_web_md')

def NBcount(questionText,answerText,contextText):
    if questionText.startswith('Who') or questionText.startswith('Whom') or questionText.startswith('Whose'):
        qType = 'PERSON'
    elif questionText.startswith('When'):
        qType = 'TIME'
    elif questionText.startswith('Where'):
        qType = 'LOCATION'
    elif questionText.startswith('How many'):
        qType = 'AMT_COUNTABLE'
    elif questionText.startswith('How much'):
        qType = 'AMT_UNCOUNTABLE'
    else:
        qType = None

    doc = nlp(answerText)
    answers = [ent for ent in doc.ents if ent.text==answerText]
    if not answers:
        aType = None
    else:
        aType = answers[0].label_
    return qType,aType
