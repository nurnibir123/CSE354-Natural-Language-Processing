import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

##################################################################
# 1. Get song lyric corpus and tokenize it


class Song:
    def __init__(self, songID, titleTokenized, lyricsTokenized):
        self.songID = songID
        self.titleTokenized = titleTokenized
        self.lyricsTokenized = lyricsTokenized


# Tokenizer from HW 1
def tokenize(sent):
    # input: a single sentence as a string.
    # output: a list of each "word" in the text
    # must use regular expressions
    pattern = r'([#@]?(?:[a-zA-Z]-)+\w|\b[A-z0-9\.\']+|[@#]*[A-z]+|\([^()]*\)|[<\[]\s*[^>]*[>\]]|[,;:?!,~=`&(){}$\^\-#\*%\/])'
    tokens = re.findall(pattern, sent)
    # <FILL IN>
    return tokens


##################################################################
# 2. Code an add1-trigram language model method

# Get random sample of 1000 lyrics

#subsetDict = dict(random.sample(songData.items(), 5000))


def createUnigramModel(dict):
    unigramModel = {}
    for key in dict.keys():
        lyrics = dict[key].lyricsTokenized
        for word in lyrics:
            if word.lower() in unigramModel:
                unigramModel[word.lower()] += 1
            else:
                unigramModel[word.lower()] = 1
    return unigramModel


# Create bigram model

def createBigramModel(dict, unigramModel):
    bigramModel = {}
    bigramModel['<OOV>'] = {}
    bigramModel['<OOV>']['<OOV>'] = 0

    for key in dict.keys():
        lyrics = dict[key].lyricsTokenized
        for i in range(len(lyrics) - 1):
            curr = lyrics[i].lower()
            next = lyrics[i + 1].lower()
            if curr not in bigramModel:
                bigramModel[curr] = {}
            if next not in bigramModel[curr]:
                bigramModel[curr][next] = 1
            if curr not in unigramModel or next not in unigramModel:
                if curr not in unigramModel and next in unigramModel:
                    if next not in bigramModel['<OOV>']:
                        bigramModel['<OOV>'][next] = 1
                    else:
                        bigramModel['<OOV>'][next] += 1
                if next not in unigramModel and curr in unigramModel:
                    if '<OOV>' not in bigramModel[curr]:
                        bigramModel[curr]['<OOV>'] = 1
                    else:
                        bigramModel[curr]['<OOV>'] += 1
                if curr not in unigramModel and next not in unigramModel:
                    bigramModel['<OOV>']['<OOV>'] += 1
            if curr in unigramModel and next in unigramModel and next in bigramModel[curr]:
                bigramModel[curr][next] += 1
    return bigramModel


# Create trigram model
def createTrigramModel(dict, unigramModel):
    trigramCounts = {}
    trigramCounts[('<OOV>', '<OOV>')] = {}
    trigramCounts[('<OOV>', '<OOV>')]['<OOV>'] = 0

    for key in dict.keys():
        lyrics = dict[key].lyricsTokenized
        for i in range(len(lyrics) - 2):
            word1 = lyrics[i].lower()
            word2 = lyrics[i + 1].lower()
            word3 = lyrics[i + 2].lower()
            if (word1, word2) not in trigramCounts:
                trigramCounts[(word1, word2)] = {}
            if word3 not in trigramCounts[(word1, word2)]:
                trigramCounts[(word1, word2)][word3] = 1
            if word1 not in unigramModel or word2 not in unigramModel or word3 not in unigramModel:
                if word1 not in unigramModel and word2 in unigramModel and word3 in unigramModel:
                    if ('<OOV>', word2) not in trigramCounts:
                        trigramCounts[('<OOV>', word2)] = {}
                    if word3 not in trigramCounts[('<OOV>', word2)]:
                        trigramCounts[('<OOV>', word2)][word3] = 1
                    else:
                        trigramCounts[('<OOV>', word2)][word3] += 1
                if word2 not in unigramModel and word1 in unigramModel and word3 in unigramModel:
                    if (word1, '<OOV>') not in trigramCounts:
                        trigramCounts[(word1, '<OOV>')] = {}
                    if word3 not in trigramCounts[(word1, '<OOV>')]:
                        trigramCounts[(word1, '<OOV>')][word3] = 1
                    else:
                        trigramCounts[(word1, '<OOV>')][word3] += 1
                if word3 not in unigramModel and word1 in unigramModel and word2 in unigramModel:
                    if '<OOV>' not in trigramCounts[(word1, word2)]:
                        trigramCounts[(word1, word2)]['<OOV>'] = 1
                    else:
                        trigramCounts[(word1, word2)]['<OOV>'] += 1
                if word1 not in unigramModel and word2 not in unigramModel and word3 not in unigramModel:
                    trigramCounts[('<OOV>', '<OOV>')]['<OOV>'] += 1
            if word1 in unigramModel and word2 in unigramModel and word3 in trigramCounts[(word1, word2)]:
                trigramCounts[(word1, word2)][word3] += 1
    return trigramCounts


# Create method that returns bigram/trigram probabilities
def calculateLMProb(prev, unigramModel, bigramModel, trigramModel, beforePrev=None):
    possibleCurrLst = [k for k in bigramModel[prev] if k != '<OOV>']
    if len(possibleCurrLst) == 0:
        vocabSize = len(unigramModel)
        unigramProbs = {}
        if prev in unigramModel:
            unigramProbs[prev] = unigramModel[prev]/vocabSize
        else:
            unigramProbs[prev] = 0
        return unigramProbs
    if beforePrev is None:
        bigramProbs = {}
        vocabSize = len(unigramModel)
        for word in possibleCurrLst:
            addOneProb = (bigramModel[prev][word] + 1)/(unigramModel[prev] + vocabSize)
            bigramProbs[word] = addOneProb
        return bigramProbs
    else:
        trigramProb = {}
        vocabSize = len(unigramModel)
        for word in possibleCurrLst:
            num = 0
            den = 0
            if (beforePrev, prev) in trigramModel and word in trigramModel[(beforePrev, prev)]:
                num = trigramModel[(beforePrev, prev)][word] + 1
            else:
                num = 1
            if beforePrev in bigramModel and prev in bigramModel[beforePrev]:
                den = bigramModel[beforePrev][prev] + vocabSize
            else:
                den = 1 + vocabSize
            addOneProb = num/den
            bigramProb = 0
            if prev in unigramModel:
                bigramProb = (bigramModel[prev][word] + 1) / (unigramModel[prev] + vocabSize)
            else:
                bigramProb = (bigramModel[prev][word] + 1) / (1 + vocabSize)
            addOneProb = (addOneProb+bigramProb)/2
            trigramProb[word] = addOneProb
        return trigramProb


##################################################################
# 3. Create adjective-specific language models

def countVowels(word):
    count = 0
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']
    for i in range(len(word)):
        if word[i].lower() in vowels:
            count += 1
    return count


def countConsts(word):
    count = 0
    vowels = ['a', 'e', 'i', 'o', 'u']
    for i in range(len(word)):
        if word[i].lower() not in vowels:
            count += 1
    return count


def getFeaturesForTokens(tokens, wordToIndex, taggedAdjs):
    # input: tokens: a list of tokens,
    # wordToIndex: dict mapping 'word' to an index in the feature list.
    # output: list of lists (or np.array) of k feature values for the given target

    wordToIndex['OOV'] = len(wordToIndex)

    num_words = len(tokens)
    featuresPerTarget = list()  # holds arrays of feature per word

    for targetI in range(num_words):
        # <FILL IN>

        adjSuffixes = ['al', 'ial', 'ical', 'able', 'ible', 'an', 'ian', 'ary', 'full', 'ic', 'ive', 'ish', 'ous', 'ile']

        featureValues = np.zeros(len(wordToIndex)+1)
        featurePrev = np.zeros(len(wordToIndex)+1)
        featureNext = np.zeros(len(wordToIndex)+1)
        featureSuffix = 0
        featureTaggedAdj = 0

        if tokens[targetI].lower() not in wordToIndex:
            featureValues.itemset(wordToIndex['OOV'], 1)
        else:
            featureValues.itemset(wordToIndex[tokens[targetI].lower()], 1)
        if targetI != 0 and targetI != (num_words-1):
            if tokens[targetI-1].lower() not in wordToIndex or tokens[targetI+1].lower() not in wordToIndex:
                if tokens[targetI-1].lower() not in wordToIndex:
                    featurePrev.itemset(wordToIndex['OOV'], 1)
                if tokens[targetI+1].lower() not in wordToIndex:
                    featureNext.itemset(wordToIndex['OOV'], 1)
                if tokens[targetI-1].lower() in taggedAdjs or tokens[targetI+1].lower() in taggedAdjs:
                    featureTaggedAdj = 200
            else:
                featurePrev.itemset(wordToIndex[tokens[targetI-1].lower()], 1)
                featureNext.itemset(wordToIndex[tokens[targetI+1].lower()], 1)

        numVowels = 0
        numConsts = 0

        if tokens[targetI] in wordToIndex:
            numVowels = countVowels(tokens[targetI])
            numConsts = countConsts(tokens[targetI])

        for suffix in adjSuffixes:
            if suffix in tokens[targetI].lower():
                featureSuffix = 100

        if tokens[targetI].lower in taggedAdjs:
            featureTaggedAdj = 200

        wordLength = len(tokens[targetI])

        featureValues = np.concatenate((featureValues,
                                        featurePrev,
                                        featureNext,
                                        [numVowels],
                                        [numConsts],
                                        [wordLength],
                                        [featureSuffix],
                                        [featureTaggedAdj]))
        featuresPerTarget.append(featureValues)

    return featuresPerTarget  # a (num_words x k) matrix


def trainAdjectiveClassifier(features, adjs):
    # inputs: features: feature vectors (i.e. X)
    #        adjs: whether adjective or not: [0, 1] (i.e. y)
    # output: model -- a trained sklearn.linear_model.LogisticRegression object
    # <FILL IN>

    X_train, X_test, y_train, y_test = train_test_split(np.array(features),
                                                        np.array(adjs),
                                                        test_size=0.1,
                                                        random_state=42)

    Cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000, 10000000]
    bestFit = None
    bestAcc = 0
    for c in Cs:
        model = LogisticRegression(C=c, penalty='l1', solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        leny = len(y_test)
        acc = np.sum([1 if (y_pred[i] == y_test[i]) else 0 for i in range(leny)]) / leny
        if acc >= bestAcc:
            bestAcc = acc
            bestFit = model
    #print('\n  Best C for test data was ', bestFit.C, ' with an accuracy of %.4f' % bestAcc, '\n')
    return bestFit


def getConllTags(filename):
    # input: filename for a conll style parts of speech tagged file
    # output: a list of list of tuples
    #        representing [[[word1, tag1], [word2, tag2]]]
    wordTagsPerSent = [[]]
    sentNum = 0
    with open(filename, encoding='utf8') as f:
        for wordtag in f:
            wordtag = wordtag.strip()
            if wordtag:  # still reading current sentence
                (word, tag) = wordtag.split("\t")
                wordTagsPerSent[sentNum].append((word, tag))
            else:  # new sentence
                wordTagsPerSent.append([])
                sentNum += 1
    return wordTagsPerSent


def generateLyrics(adj, adjToLyrics):
    songIDList = adjToLyrics[adj]
    songDict = {}
    for songID in songIDList:
        songDict[songID] = songData[songID]
    unigramModel = createUnigramModel(songDict)
    subsetUnigramModel = {k: v for k, v in unigramModel.items() if v > 2}
    bigramModel = createBigramModel(songDict, subsetUnigramModel)
    trigramModel = createTrigramModel(songDict, subsetUnigramModel)
    lyrics = ['<s>']
    secWordProb = calculateLMProb('<s>', subsetUnigramModel, bigramModel, trigramModel)
    words = list(secWordProb.keys())
    probs = list(secWordProb.values())
    probs = np.array(probs)
    probs = probs/probs.sum()
    secondWord = np.random.choice(words, 1, p=probs)
    lyrics.append(secondWord[0])
    for i in range(30):
        nextWordProb = calculateLMProb(lyrics[-1], subsetUnigramModel, bigramModel, trigramModel, lyrics[-2])
        words = list(nextWordProb.keys())
        probs = list(nextWordProb.values())
        probs = np.array(probs)
        probs = probs/probs.sum()
        nextWord = np.random.choice(words, 1, p=probs)
        if nextWord[0] == '</s>':
            lyrics.append(nextWord[0])
            break
        lyrics.append(nextWord[0])
    lyrics.append('</s>')
    return lyrics


if __name__ == '__main__':
    # Read csv file using pandas
    data = pd.read_csv("songdata.csv")
    artists = data['artist']
    songTitles = data['song']
    songLyrics = data['text']

    songIDList = []
    titlesTokenized = []
    lyricsTokenized = []

    # Create a dictionary to store Song objects
    songData = {}

    for i in range(len(artists)):
        artistsID = artists[i].replace(' ', '_').lower()
        songTitlesID = songTitles[i].replace(' ', '_').lower()
        songID = artistsID + '-' + songTitlesID
        lyricsTokens = '<s>' + songLyrics[i] + '<s>'
        lyricsTokens = lyricsTokens.replace('\n', '<newline>')
        songData[songID] = Song(songID, tokenize(songTitles[i]), tokenize(lyricsTokens))

    print('\nStage One Checkpoint: \n')
    checkpointOne = ['abba-burning_my_bridges',
                     'beach_boys-do_you_remember?',
                     'avril_lavigne-5,_4,_3,_2,_1_(countdown)',
                     'michael_buble-l-o-v-e']

    for songID in checkpointOne:
        print(songData[songID].titleTokenized)
        print(songData[songID].lyricsTokenized, '\n')

    subsetDict = {k: songData[k] for k in list(songData)[:5000]}

    totalWordCount = createUnigramModel(subsetDict)

    wordCount = {k: v for k, v in totalWordCount.items() if v > 2}

    bigramCounts = createBigramModel(subsetDict, wordCount)

    trigramCounts = createTrigramModel(subsetDict, wordCount)

    print('\nStage Two Checkpoint: \n')

    print(calculateLMProb('love', wordCount, bigramCounts, trigramCounts, 'i')['you'], '\n')
    print(calculateLMProb('midnight', wordCount, bigramCounts, trigramCounts)['special'], '\n')
    print(calculateLMProb('very', wordCount, bigramCounts, trigramCounts)['special'], '\n')
    print(calculateLMProb('very', wordCount, bigramCounts, trigramCounts, 'something')['special'], '\n')
    print(calculateLMProb('very', wordCount, bigramCounts, trigramCounts, 'something')['funny'], '\n')

    taggedSents = getConllTags('daily547.conll')

    wordToIndex = set()  # maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)  # splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndex |= set([w.lower() for w in words])  # union of the words into the set
    # print("  [Read ", len(taggedSents), " Sentences]")
    # turn set into dictionary: word: index
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    # Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    # print("  [Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex, taggedSents))
            sentYs.append([1 if t == 'A' else 0 for t in tags])

    X = [j for i in sentXs for j in i]
    y = [j for i in sentYs for j in i]

    # Train the model.
    # print("  [Training the model]")
    adjClassifier = trainAdjectiveClassifier(np.array(X), np.array(y))
    # print("  [Done]")

    songTitleTokens = []
    for songID in songData:
        for token in songData[songID].titleTokenized:
            songTitleTokens.append(token.lower())

    # print(len(songTitleTokens))

    taggedAdjs = []
    for lst in taggedSents:
        taggedAdjs += [pair[0].lower() for pair in lst if pair[1] == 'A']

    songTitleFeatures = getFeaturesForTokens(songTitleTokens, wordToIndex, taggedAdjs)

    adjPredictedIndexes = adjClassifier.predict(songTitleFeatures)

    # print(adjPredictedIndexes)

    adjPredicted = [songTitleTokens[i] for i in range(len(adjPredictedIndexes)) if adjPredictedIndexes[i] == 1]
    adjPredicted = list(dict.fromkeys(adjPredicted))

    # print(adjPredicted)

    adjToLyrics = {}

    for adj in adjPredicted:
        if adj in adjToLyrics:
            continue
        count = 0
        songIdLst = []
        for songID in songData:
            for token in songData[songID].titleTokenized:
                if adj == token.lower():
                    count += 1
                    songIDList.append(songID)
        if count >= 10:
            if adj not in adjToLyrics:
                adjToLyrics[adj] = songIDList

    print('\nStage Three Checkpoint: \n')
    checkpointThree = ['good', 'happy', 'afraid', 'red', 'blue']
    for adj in checkpointThree:
        if adj in adjToLyrics:
            print('First 10 artist-songs for ', adj, ': ', adjToLyrics[adj][:10], '...')
            print('Lyrics for ', adj, ': ', generateLyrics('good', adjToLyrics))
            print('Lyrics for ', adj, ': ', generateLyrics('good', adjToLyrics))
            print('Lyrics for ', adj, ': ', generateLyrics('good', adjToLyrics), '\n')
        else:
            print(adj, ' not classified as adjective', '\n')
