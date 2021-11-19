possessives = ['his','her','their','its']

import coreferee,spacy

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

#doc = nlp('Thomas Edison was an American inventor. He invented the lightbulb. His other inventions include the phonograph and the motion picture. He married his wife in 1860; they had four children.') 
#doc._.coref_chains.print()

def patchList(L,span,filler):
    return L[:span[0]] + filler + L[span[1]:]


def replaceInDoc(doc):
    phraseDic = {p.root:p for p in doc.noun_chunks}
    newList = list(doc[:])
    for i in range(len(doc)-1,-1,-1):
        token = doc[i]
        referents = doc._.coref_chains.resolve(token)
        if not referents:
            continue

        filler = []
        for (j,ref) in enumerate(referents):
            if j > 0:
                filler.append('and ')

            if referents[0] in phraseDic:
                filler.extend(list(phraseDic[ref]))
            else:
                filler.extend([ref])

            if token.text.lower() in possessives:
                filler.append("~'s ")

        if token in phraseDic:
            patchSpan = (phraseDic[token].start,phraseDic[token].end)
        else:
            patchSpan = (i,i+1)

        if doc[patchSpan[1]-1].whitespace_:
            filler.append('~ ')

        newList = patchList(newList,patchSpan,filler)

    msg = ''
    for token in newList:
        if type(token) == spacy.tokens.Token:
            msg += token.text + token.whitespace_
        else:
            if token[0] == '~':
                if msg[-1] == ' ':
                    msg = msg[:-1]
                msg += token[1:]
            else:
                msg += token
    return msg


def cleanNewlines(rawText):
    nText = ''
    for i in range(len(rawText)-1):
        char,nextChar = rawText[i],rawText[i+1]
        if char != '\n' and nextChar == '\n':
            nText += char + ('.' if char != '.' else '') + ' '
        else:
            nText += char
    return nText.replace('\n','')

def preprocess(rawText):
    rawText = cleanNewlines(rawText)
    doc = nlp(rawText)
    return replaceInDoc(doc)

# if __name__ == '__main__':
#     with open('lincoln.txt') as f:
#         rawText = f.read()
#     rawText = preprocess(rawText)
#     with open('lincolnCoref.txt','w') as f:
#         f.write(rawText)
#     print('Wrote preprocessed text to lincolnCoref.txt')

