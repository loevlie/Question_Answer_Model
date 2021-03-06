#!/usr/bin/env python3

from numpy import append
import spacy
import inflect
import sys
from preprocess import preprocess
import random
from spacy import displacy
import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_md')
joiningPOS = ['PUNCT','CCONJ','SCONJ']
questions = []
def fixedSimilarity(span1,span2):
    if type(span1) == spacy.tokens.Token or len(span1) <= 1:
        comp1 = span1
    else:
        fixed1 = ' '.join(token.text for token in span1 if not token.is_stop)
        comp1 = nlp(fixed1) if fixed1 else span1

    if type(span2) == spacy.tokens.Token or len(span2) <= 1:
        comp2 = span2
    else:
        fixed2 = ' '.join(token.text for token in span2 if not token.is_stop)
        comp2 = nlp(fixed2) if fixed2 else span2
    
    return comp1.similarity(comp2)

def subTree(word,signposts):
    L = [child for child in word.children if child not in signposts]
    for child in word.children:
        if child in signposts:
            continue
        L.extend(subTree(child,signposts))
    return L

def displayStructure(senseObj):
    if not senseObj.subject or not senseObj.predicate:
        return ''
    msg = senseObj.subject.displayTree()
    msg += ' -- '
    if senseObj.gullets:
        gulletStr = '<' + '> <'.join(gullet.displayTree() for \
                gullet in senseObj.gullets) + '> '
    else:
        gulletStr = ''
    msg += gulletStr + (senseObj.gulletText if senseObj.gulletText else senseObj.rootToken.text)
    msg += ' -> ' + \
           (senseObj.predicate.displayTree() if senseObj.predicate else '[NO PREDICATE]')
    return msg
    
def splitIntoClauses(doc):
    if not doc:
        return []
    
    root = doc[:].root
    
    subj = [c for c in root.children if c.dep_ == 'nsubj' or c.dep_ == 'nsubjpass' or c.dep_ == 'attr']
    if not subj:
        return [doc]
    
    primaryVerbs = []
    if root.pos_ == 'VERB' or root.pos_=='AUX':
        primaryVerbs.append(root)
    
    primaryVerbs += [c for c in root.children if c.pos_ == 'VERB' or c.pos_ == 'AUX']
    
    for verb in primaryVerbs:
        if verb == root:
            continue
        dropSpan = doc[min(root.i,verb.i):max(root.i,verb.i)]
        drop = True
        for word in dropSpan:
            if word.pos_ in joiningPOS:
                drop = False
        if drop:
            primaryVerbs.remove(verb)
    if len(primaryVerbs) <= 1:
        return [doc]
    clauses = []
    for verb in primaryVerbs:
        
        startSpan,endSpan = verb.i,verb.i
        for child in subTree(verb,primaryVerbs):
            if child.dep_ == 'punct':
                continue
            if child.i < startSpan:
                startSpan = child.i
            if child.i > endSpan:
                endSpan = child.i
        candidateClause = doc[startSpan:endSpan+1]
        if candidateClause.text.strip():
            clauses.append(candidateClause)
    nClauses = []
    
    for clause in clauses:
        if len(clause) < 3:
            continue
        clauseSubj = [c for c in clause.root.children if c.dep_ == 'nsubj' or c.dep_ == 'nsubjpass' or c.dep_ =='attr']
        if not clauseSubj and clause[0].pos_ not in joiningPOS:
            subjToken = subj[0]
            clause = nlp(subjToken.text + ' ' + clause.text)
        if clause[-1].pos_ == 'CCONJ':
            clause = clause[:-1]
        if clause[-1].pos_ == 'PUNCT':
            clause = clause[:-1]
        if clause:
            nClauses.append(clause)
    return nClauses

def splitClausesFully(doc):
    s = splitIntoClauses(doc)
    while True:
        newS = []
        for i in s:
            for j in splitIntoClauses(nlp(i.text)):
                if not any(k.text == j.text for k in newS):
                    newS.append(j[:])
        if len(newS) <= len(s):
            final = []
            for sent in s:
                final.append(sent.text)
            return final
        s = newS

def generate_questions(corpus):

    p = inflect.engine()
    corpus = corpus.strip(".")
    sentences = corpus.split(". ")
    when_question_count = 0
    where_question_count = 0
    who_question_count=0
    import math
    Average_count = math.ceil(len(sentences)/2)

    for sentence in sentences:
        doc = nlp(sentence)
        cs = splitClausesFully(doc)

        for c in cs:
            doc = nlp(c)
            displacy.render(doc, style="ent")
            best_question = ""
            best_question_score = 0
            named_entities = {}
            for ent in doc.ents:
                named_entities[ent.text] = ent.label_
            all_words = []
            for tok in doc:
                all_words.append(tok)
            #----------------------------------------------------------- Who Questions -----------------------------------------------------------
            for chunk in doc.noun_chunks:
            
                if chunk.text in named_entities and named_entities[chunk.text] == "PERSON":
                    q = "Who is " + chunk.text +"?"
                    similar = False
                    if q in questions:
                        similar = True
                    if similar == False:
                        sim = nlp(q).similarity(doc)
                        if sim > best_question_score:
                            best_question = q
                            best_question_score = sim
                    
                if chunk.root.dep_ == "nsubj" or chunk.root.dep_ == "nsubjpass":
                    rest_of_sentence = ""
                    start_index = list(chunk)[-1].i + 1
                    for i in range(start_index, len(all_words)):
                        if (all_words[i].dep_ == "ROOT" and all_words[i].pos_ == "VERB") and (all_words[i].morph.get("VerbForm")[0] == "Fin" and all_words[i].morph.get("Tense")[0] == "Pres"):
                            rest_of_sentence += p.plural(all_words[i].lemma_) + " "
                            
                        elif all_words[i].text == "have":
                            rest_of_sentence += "has "
                        
                        elif all_words[i].text == "are":
                            rest_of_sentence += "is "
                
                        elif all_words[i].text == "were":
                            rest_of_sentence += "was "

                        else:
                            rest_of_sentence += all_words[i].text + " "
        
                    rest_of_sentence = rest_of_sentence.strip()
                    q = "Who " + rest_of_sentence + "?"
                    similar = False
                    if q in questions:
                        similar = True
                    if similar == False:
                        sim = nlp(q).similarity(doc)
                        if sim > best_question_score and who_question_count<=Average_count:
                            best_question = q
                            best_question_score = sim
                            who_question_count+=1


            #----------------------------------------------------------- Where Questions -----------------------------------------------------------
            dobj = ""
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ == "nsubj":

                    subject = chunk

                    if chunk.text not in named_entities and chunk.text != "I":
                        subject_text = chunk.text.lower()
                    else:
                        subject_text = chunk.text

                    root = chunk.root.head
                    if root.pos_ == "VERB" and (root.morph.get("VerbForm")[0] == "Inf" or root.morph.get('Tense')[0] == "Pres"):
                        if len(list(root.lefts)) > 2 and list(root.lefts)[-1].pos_ == "AUX" and list(root.lefts)[-2].pos_ == "AUX":
                            aux1 = list(root.lefts)[-2].text
                            aux2 = list(root.lefts)[-1].text
                            verb = root.text
                            phrase = "Where " + aux1 + " " + subject_text + " " + aux2 + " " + verb

                        elif list(root.lefts)[-1].pos_ == "AUX":
                            aux = list(root.lefts)[-1].text
                            verb = root.text
                            phrase = "Where " + aux + " " + subject_text + " " + verb

                        else:
                            verb = root.lemma_
                            if subject_text == "I":
                                phrase = "Where do " + subject_text + " " + verb
                            else:
                                phrase = "Where does " + subject_text + " " + verb
                        
                    elif root.pos_ == "VERB" and root.morph.get('Tense')[0] == "Past":
                        if list(root.lefts)[-1].pos_ == "AUX":
                            aux = list(root.lefts)[-1].text
                            verb = root.text
                            phrase = "Where " + aux + " " + subject_text + " " + verb
                        else:
                            verb = root.lemma_
                            phrase = "Where did " + subject_text + " " + verb

                    
                    elif root.pos_ == "AUX" and (root.morph.get("VerbForm")[0] == "Inf" or root.morph.get('Tense')[0] == "Pres"):
                        if list(root.lefts) and list(root.lefts)[-1].pos_ == "AUX" :
                            verb = ""
                            phrase = "Where " + list(root.lefts)[-1].text + " " + root.text + " " + subject_text
                        else:
                            verb = ""
                            phrase = "Where " + root.text + " " + subject_text 
                            
                    elif root.pos_ == "AUX" and root.morph.get('Tense')[0] == "Past":
                        if list(root.lefts) and list(root.lefts)[-1].pos_ == "AUX":
                            verb = ""
                            phrase = "Where " + list(root.lefts)[-1].text + " " + root.text + " " + subject_text 
                        else:
                            verb = ""
                            phrase = "Where " + root.text + " " + subject_text

                if chunk.root.dep_ == "dobj":
                    dobj = chunk.text + " "

                b = False
                if chunk.root.dep_ == "pobj":
                    for entity in named_entities:
                        if chunk.text in entity:
                            if named_entities[entity] == "TIME" or named_entities[entity] == "DATE":
                                b = True
                                break
                    if b:
                        continue

                    if chunk.root.head.text == "from" and (root.pos_ == "AUX" or root.pos_ == "VERB"):
                        prep = "from "
                    elif chunk.root.head.text == "to" and (root.pos_ == "AUX" or root.pos_ == "VERB"):
                        prep = "to "
                    else:
                        prep = ""
                    
                    prep_phrase = chunk

                    rest_of_sentence = ""
                    start_index = list(chunk)[-1].i + 1
                    for i in range(start_index, len(all_words)):
                        rest_of_sentence += all_words[i].text + " "
                    rest_of_sentence = rest_of_sentence.strip()
                    if prep_phrase.root.head.head == subject.root.head:
                        q = (phrase + " " +  dobj + prep + rest_of_sentence).strip() + "?"
                        similar = False
                        if q in questions:
                            similar = True
                        if similar == False:
                            sim = nlp(q).similarity(doc)
                            if sim > best_question_score and where_question_count<=2:
                                best_question = q
                                best_question_score = sim
                                where_question_count+=1


            # ----------------------------------------------------------- When Questions -----------------------------------------------------------
            dobj = ""
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ == "nsubj":

                    subject = chunk

                    if chunk.text not in named_entities and chunk.text != "I":
                        subject_text = chunk.text.lower()
                    else:
                        subject_text = chunk.text

                    root = chunk.root.head
                    if root.pos_ == "VERB" and (root.morph.get("VerbForm")[0] == "Inf" or root.morph.get('Tense')[0] == "Pres"):
                        if len(list(root.lefts)) > 2 and list(root.lefts)[-1].pos_ == "AUX" and list(root.lefts)[-2].pos_ == "AUX":
                            aux1 = list(root.lefts)[-2].text
                            aux2 = list(root.lefts)[-1].text
                            verb = root.text
                            phrase = "When " + aux1 + " " + subject_text + " " + aux2 + " " + verb

                        elif list(root.lefts)[-1].pos_ == "AUX":
                            aux = list(root.lefts)[-1].text
                            verb = root.text
                            phrase = "When " + aux + " " + subject_text + " " + verb

                        else:
                            verb = root.lemma_
                            if subject_text == "I":
                                phrase = "When do " + subject_text + " " + verb
                            else:
                                phrase = "When does " + subject_text + " " + verb
                        
                    elif root.pos_ == "VERB" and root.morph.get('Tense')[0] == "Past":
                        if list(root.lefts) and list(root.lefts)[-1].pos_ == "AUX":
                            aux = list(root.lefts)[-1].text
                            verb = root.text
                            phrase = "When " + aux + " " + subject_text + " " + verb
                        else:
                            verb = root.lemma_
                            phrase = "When did " + subject_text + " " + verb
        
                    elif root.pos_ == "AUX" and (root.morph.get("VerbForm")[0] == "Inf" or root.morph.get('Tense')[0] == "Pres"):
                        if list(root.lefts) and list(root.lefts)[-1].pos_ == "AUX" :
                            verb = ""
                            phrase = "When " + list(root.lefts)[-1].text + " " + root.text + " " + subject_text
                        else:
                            verb = ""
                            phrase = "When " + root.text + " " + subject_text 
                            
                    elif root.pos_ == "AUX" and root.morph.get('Tense')[0] == "Past":
                        if list(root.lefts) and list(root.lefts)[-1].pos_ == "AUX":
                            verb = ""
                            phrase = "When " + list(root.lefts)[-1].text + " " + root.text + " " + subject_text 
                        else:
                            verb = ""
                            phrase = "When " + root.text + " " + subject_text

                if chunk.root.dep_ == "dobj":
                    dobj = chunk.text + " "

                if chunk.root.dep_ == "pobj":
                    if chunk.root.head.text == "on" or chunk.root.head.text == "in" or chunk.root.head.text == "at":
                        b = False
                        if chunk.root.dep_ == "pobj":
                            for entity in named_entities:
                                if chunk.text in entity:
                                    if named_entities[entity] != "TIME" and named_entities[entity] != "DATE":
                                        b = True
                                        break
                            
                            if b:
                                continue
                            
                            rest_of_sentence = ""
                            start_index = root.i + 1
                            end_index = chunk.root.head.i
                            for i in range(start_index, end_index):
                                rest_of_sentence += all_words[i].text + " "
                            rest_of_sentence = rest_of_sentence.strip()
                            if prep_phrase.root.head.head == subject.root.head:
                                q = (phrase + " " +  dobj + rest_of_sentence).strip() + "?"
                                similar = False
                                if q in questions:
                                    similar = True
                                if similar == False:
                                    sim = nlp(q).similarity(doc)
                                    if sim > best_question_score and when_question_count<=Average_count:
                                        best_question = q
                                        best_question_score = sim
                                        when_question_count+=1
                        else:
                            continue
                    else:
                        continue
            if best_question == '':
                continue
            questions.append(best_question)
    return questions

if __name__ == '__main__':
    corpus = open(sys.argv[1], "r",encoding='ISO-8859-1').read()
    question_n = int(sys.argv[2])
    corpus = preprocess(corpus,True)
    if corpus:
        generated = generate_questions(corpus)
        if question_n > len(generated):
            generated_f = random.sample(generated, k_f=len(generated))
            generated_f.append("Question Limit Exceeded.")
        else:
            generated_f = random.sample(generated, k=question_n)     
        for question in generated_f:
            if question == "":
                question = random.choice(generated)
            print(question)
    # print(generate_questions())
