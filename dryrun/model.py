
def ruleBasedModel(featureVectors):
    import random
    fullDict = {}
    for i in featureVectors:
        fullDict.update(i)

    greatCandidates,okayCandidates = {},{}
    for ans in fullDict:
        vec = fullDict[ans]
        if (vec[1] > 0.8 and vec[2] > 0.5) or (vec[2] > 0.8 and vec[1] > 0.5):
            fixedScore = int(10 * (vec[1] * vec[2]))/10.
            if vec[0] == 1 and vec[6] == 0:
                fixedScore -= 0.5
            
            if fixedScore not in greatCandidates:
                greatCandidates[fixedScore] = [ans]
            else:
                greatCandidates[fixedScore].append(ans)
        elif vec[2] > 0.2 and vec[4] < 0.5:
            okayCandidates[ans] = vec

    if greatCandidates:
        bestBatch = greatCandidates[max(greatCandidates)]
        #print('Great candidates: ',bestBatch)
        if len(bestBatch) == 1:
            return bestBatch[0].text
        chainLengths = [fullDict[ans][5] for ans in bestBatch]
        finalAnswers = [ans for i,ans in enumerate(bestBatch) if chainLengths[i] == min(chainLengths)]
        if len(finalAnswers) == 1:
            return finalAnswers[0].text
        fineGrained = [fullDict[ans][1] + fullDict[ans][2] for ans in bestBatch]
        finalAnswers = [ans for i,ans in enumerate(bestBatch) if fineGrained[i] == max(fineGrained)]
        if len(finalAnswers) == 1:
            return finalAnswers[0].text
        return random.choice(finalAnswers).text # oh, just give up

    bestScore,bestAns = 0,None
    for ans in okayCandidates:
        vec = okayCandidates[ans]
        score = vec[1] + vec[2]
        if vec[0] == 1 and vec[6] == 0:
            score -= 0.5
        if vec[5] > 8:
            score -= 0.4
        elif vec[5] > 4:
            score -= 0.2

        #print('Okay candidate: ',ans,score)
        if score > bestScore:
            bestScore = score
            bestAns = ans
    if bestAns:
        return bestAns.text

    return random.choice(fullDict)
        
        
