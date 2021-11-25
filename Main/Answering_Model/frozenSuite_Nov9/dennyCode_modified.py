# Imports 

import numpy as np 
import spacy

nlp = spacy.load('en_core_web_md')

def find_similar_sentences(raw_text,question,length=5):
    # Break the text into sentences
    questionEnts = {e.root:e for e in question.ents}
    question_nostop = nlp(' '.join(token.text for token in question if not token.is_stop))
    
    answer = raw_text
    
    origSentences,sims = [],[]
    for s in answer.sents:
        if not s.text.strip():
            continue
        
        sentenceFixed = ' '.join(token.text.strip() for token in s\
                                   if not token.is_stop)
        sentence_nostop = nlp(sentenceFixed)
        sim = sentence_nostop.similarity(question_nostop)

        answerEnts = {e.root:e for e in s.ents}
        if questionEnts:
            for qRoot in questionEnts:
                if len(questionEnts[qRoot]) > 1 and any(answerEnts[aRoot].text == questionEnts[qRoot].text for aRoot in answerEnts):
                    sim += 0.10
                elif any(aRoot.lemma_ == qRoot.lemma_ for aRoot in answerEnts):
                    sim += 0.05
                else:
                    sim -= 0.05
        
        sims.append(sim)
        origSentences.append(s)

    simVec = np.array(sims)
    bestIndices = np.argsort(simVec)[(len(simVec)-length):]

    outDict = {}
    for i in bestIndices:
        outDict[simVec[i]] = origSentences[i]
        #print(origSentences[i],simVec[i])
        
    return outDict

if __name__ == '__main__':
    Answer_File = 'messi.txt'
    with open(Answer_File,'r',encoding="ISO-8859-1") as f:
        rawText = f.read()

    rawText = rawText.replace('\n','.')
    
    while True:
        question = input('Enter a question. >')
        if not question:
            break
        print(find_similar_sentences(rawText,question))
