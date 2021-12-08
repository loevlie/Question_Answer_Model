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

def questionHasAnswer(listOfAnswers,contextText):
    context = nlp(contextText)
    phrases = list(context.noun_chunks)
    ents = context.ents

    for ans in listOfAnswers:
        if any(ent.text == ans for ent in ents) or any(p.text == ans for p in phrases):
            return True
    
    return False

QAfile = '..\\..\\Data\\Question_Answer_Dataset_v1.2\\S09\\question_answer_pairs.txt'
with open(QAfile,encoding='ISO-8859-1') as f:
    lines = f.readlines()[1:]

docDict = {}
import preprocess

for num1 in range(1,6):
    for num2 in range(1,11):
        with open('../../Data/Question_Answer_Dataset_v1.2/S09/data/set{}/a{}.txt'.format(num1,num2),encoding='ISO-8859-1') as f:
            volta = f.read()
        volta = preprocess.preprocess(volta)
        voltaDoc = nlp(volta)
        docDict['data/set{}/a{}'.format(num1,num2)] = voltaDoc

correct,wrong = 0,0
import binaryAnswers,QAfeatures

for line in lines:
    title,q,a,d1,d2,filepath = line.split('\t')
    filepath = filepath.strip()
    
    fixed_a = ''.join(char.lower() for char in a if char.isalnum())
    if fixed_a != 'yes' and fixed_a != 'no':
        continue

    print('\nQUESTION: ' + q)
    QS = QAfeatures.QuestionSense(q)
    adoc = docDict[filepath]

    ourAnswer = binaryAnswers.runThroughSentences(QS,adoc)
    ourAnswer = 'yes' if ourAnswer==True else 'no'

    if ourAnswer == fixed_a:
        correct +=1
    else:
        wrong += 1

    print(fixed_a,ourAnswer,correct/(correct+wrong))

print(correct/(correct + wrong))
