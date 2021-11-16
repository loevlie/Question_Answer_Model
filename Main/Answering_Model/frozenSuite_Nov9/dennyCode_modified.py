# Imports 

import numpy as np 
import spacy

nlp = spacy.load('en_core_web_md')

def find_similar_sentences(raw_text,question,length=5):
    # Break the text into sentences
    question = nlp(question)
    question_nostop = nlp(' '.join(token.text for token in question if not token.is_stop))
    
    answer = nlp(raw_text)
    
    sentences = []
    origSentences = []
    for s in answer.sents:
        if not s.text.strip():
            continue
        sentences.append(' '.join(token.text.strip() for token in s\
                                   if not token.is_stop))
        origSentences.append(s)
    
    #print(len(sentences))
    highly_similar_sentences = {}
    sims = []
    
    for sentenceText in sentences:
        sentence_nostop = nlp(sentenceText)
        sim = sentence_nostop.similarity(question_nostop)
        sims.append(sim)
        
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
        #print(find_similar_sentences(rawText,question))
