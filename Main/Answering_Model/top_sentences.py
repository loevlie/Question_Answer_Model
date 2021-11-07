# Imports 

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from collections import defaultdict

# Example of an Answer File from the Data Folder
Answer_File = '../Data/Question_Answer_Dataset_v1.2/S10/question_answer_pairs.txt'

with open(Answer_File,'r',encoding="ISO-8859-1") as f:
    Questions = f.read().split('\n')

Data = {key:[] for key in Questions[0].split('\t')}

keys = list(Data.keys())

for row in range(1,len(Questions)):
    data_point = Questions[row].split('\t')
    if len(data_point)>1:
        for i in range(len(data_point)):
            Data[keys[i]].append(data_point[i])

df = pd.DataFrame(Data)

nlp = spacy.load('en_core_web_md')
def is_token_allowed(token):
    '''
        Only allow valid tokens which are not stop words
        and punctuation symbols.
    '''
    if (not token or not str(token).strip() or token.is_stop or token.is_punct):
        return False
    return True

def preprocess_token(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()

def Get_Tokens(file_path):
    with open('../Data/Question_Answer_Dataset_v1.2/'+'S10/'+file_path+'.txt','r') as g:
        text = g.read()
    Article = nlp(text)
    return Article

def Get_Token_Sentences(file_path):
    with open('../Data/Question_Answer_Dataset_v1.2/'+'S10/'+file_path+'.txt','r') as g:
        text = g.read()
    Article = nlp(text)
    complete_filtered_tokens = [token for token in Article if token]
    return complete_filtered_tokens

Articles = defaultdict(list)

for i,path in enumerate(df['ArticleFile']):
    if df['ArticleTitle'][i] not in list(Articles.keys()):
        Articles[df['ArticleTitle'][i]] = Get_Tokens(path)


def find_similar_sentences(raw_text,question):
    # Break the text into sentences
    nlp.add_pipe('sentencizer') # updated
    
    question = nlp(question)
    sentences = [sent.text.strip() for sent in raw_text.sents]
    highly_similar_sentences = {}
    sims = []
    for i,sentence in enumerate(sentences):
        if sentence == '':
            continue
        sentence = nlp(sentence)
        sentence_no_stop_words = nlp(' '.join([str(t) for t in sentence if not t.is_stop]))
        question_no_stop_words = nlp(' '.join([str(t) for t in question if not t.is_stop]))
        
        sim = sentence_no_stop_words.similarity(question_no_stop_words)
        #if sim >= 0.7:
            #highly_similar_sentences.update({i:sim})
        
        sims.append(sim)
    highly_similar_sentences.update({np.argmax(sims):np.max(sims)})
            
    return np.array(sentences)[list(highly_similar_sentences.keys())],highly_similar_sentences

def get_similar_sentences(df,Articles,length=10):
    sol = defaultdict(list)
    for i in range(length):
        print(i)
        similar_sentence, scores = find_similar_sentences(Articles[df.iloc[i]['ArticleTitle']],df.iloc[i]['Question'])
        sol['Questions'].append(df.iloc[i]['Question'])
        sol['Similar_sentence'].append(similar_sentence)
        sol['Similarity_Score'].append(scores)
    return sol

sol = get_similar_sentences(df,Articles)
print(sol)