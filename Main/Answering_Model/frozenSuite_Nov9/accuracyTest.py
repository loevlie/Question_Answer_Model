from Utils import get_features
from model import ruleBasedModel
import spacy

import warnings
warnings.filterwarnings('ignore')

import numpy

vectors = numpy.load('X_test_full.npy')
answers = numpy.load('y_test_full.npy')

correct = 0

confCorrect,confTotal = 0,0

for i in range(vectors.shape[0]):
    fakeDict = {}
    for j in range(vectors.shape[1]):
        vec = vectors[i,j,:]
        if vec[0] == -1:
            continue
        fakeDict[j] = vec

    ourAnswer = ruleBasedModel(fakeDict)
    actAnswer = numpy.argmax(answers[i,:])

    if type(ourAnswer) == int:
        truthVal = (ourAnswer == actAnswer)
    elif type(ourAnswer) == list:
        truthVal = (actAnswer in ourAnswer)
    else:
        truthVal = False

    #truthVal = (ourAnswer==actAnswer) if type(ourAnswer)==int else (actAnswer in ourAnswer)

    if truthVal:
        correct += 1

    if ourAnswer != None:
        confTotal += 1
        if truthVal:
            confCorrect += 1

print(100 * correct/vectors.shape[0])
print(100 * confCorrect/confTotal)
