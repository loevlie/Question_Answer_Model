import QAfeatures
import numpy

def ruleBasedModel(fullDict):
    import random

    #print('Candidates: ')
    
    greatCandidates,okayCandidates = {},{}
    doWeCareAboutVerbs = any(fullDict[key][2] > 0.2 for key in fullDict)
    for ans in fullDict:
        vec = fullDict[ans]
        #print(ans.text,vec)
        if (vec[1] > 0.8 and vec[2] > 0.5) or (vec[2] > 0.8 and vec[1] > 0.5):
            fixedScore = int(10 * (vec[1] * vec[2]))/10.
            if vec[0] == 1 and vec[6] < 0.5:
                fixedScore -= 0.5
            if vec[4] >= 0.5:
                fixedScore -= 0.3
                
            if fixedScore <= 0.4:
                continue
            
            if fixedScore not in greatCandidates:
                greatCandidates[fixedScore] = [ans]
            else:
                greatCandidates[fixedScore].append(ans)
        elif vec[4] < 0.5:
            if doWeCareAboutVerbs:
                if vec[2] > 0.2:
                    okayCandidates[ans] = vec
            else:
                okayCandidates[ans] = vec

    if greatCandidates:
        bestBatch = greatCandidates[max(greatCandidates)]
        #print('Great candidates: ' + '; '.join(i.text for i in bestBatch))
        if len(bestBatch) == 1:
            return bestBatch[0]
        chainLengths = [fullDict[ans][5] for ans in bestBatch]
        finalAnswers = [ans for i,ans in enumerate(bestBatch) if chainLengths[i] == min(chainLengths)]
        if len(finalAnswers) == 1:
            return finalAnswers[0]
        fineGrained = [fullDict[ans][1] + fullDict[ans][2] for ans in bestBatch]
        finalAnswers = [ans for i,ans in enumerate(bestBatch) if fineGrained[i] == max(fineGrained)]
        if len(finalAnswers) == 1:
            return finalAnswers[0]
        #print('Warning: we have a tie between great answers')
        return random.choice(finalAnswers) # oh, just give up

    bestScore,bestAns = 0,None
    namedEntityFlag = vec[0] == 1 and \
                      any(okayCandidates[k][6] == 1 for k in okayCandidates)
    for ans in okayCandidates:
        vec = okayCandidates[ans]
        score = vec[1] + vec[2]

        if namedEntityFlag and vec[6] == 0:
            continue
        elif vec[0] == 1 and vec[6]==0:
            score -= 0.5
        elif vec[0] == 3 and vec[6] != 0:
            score -= 0.05
        
        if vec[5] > 8:
            score -= 0.3
        elif vec[5] > 4:
            score -= 0.1
        elif vec[5] > 2:
            score -= 0.05

        score -= 0.12 * vec[9]

        score *= vec[7]

        #print('Okay candidate: ',ans,score)
        if score > bestScore:
            bestScore = score
            bestAns = ans
        elif score == bestScore:
            if bestAns == None:
                bestAns = ans
            else:
                if type(bestAns) == list:
                    bestAns.append(ans)
                else:
                    bestAns = [bestAns,ans]
        
    if bestAns:
        if type(bestAns) == list:
            tiebreakList = [fullDict[a][8] for a in bestAns]
            bestAns = [a for i,a in enumerate(bestAns) if tiebreakList[i] == min(tiebreakList)]
            if len(bestAns) == 1:
                return bestAns[0]
            else:
                #print('Warning: we have a tie between okay answers')
                return bestAns
        else:  
            return bestAns

    #print('Warning: we have not found even a somewhat passable answer')
    return None

def neuralNetModel(fullDict):
    import NN_Model_Use
    import numpy as np
    
    answers = np.array(list(fullDict.keys()))
    vectors = np.array(list(fullDict.values()))
    Padding = np.ones((30-vectors.shape[0],vectors.shape[1]))
    if len(vectors) < 30:
        vectors = np.concatenate((vectors,Padding))
        #vectors.extend(Padding*(30-len(vectors)))
        answers =np.append(answers,np.array(['NA']*(30-len(answers))))
        #answers.extend(['NA']*(30-len(answers)))
    print('Ans length: {}'.format(len(answers)))
    
    random_indices = np.arange(30)
    np.random.shuffle(random_indices)
    vectors = vectors[random_indices]
    answers = answers[random_indices]
    print(vectors)
    print()
    print(answers)
    #flattenedVec = numpy.hstack(vectors)
    #flattenedVec = numpy.hstack((flattenedVec,-1*numpy.ones(30*10 - len(flattenedVec))))
    #print(
    ansIndex = NN_Model_Use.XGBoost_Answer(vectors)
    print(ansIndex)


    ans = answers[ansIndex]
    print(ans)
    if type(ans) == str:
        return None
    return answers[ansIndex]
    
        
        
