#!/usr/bin/python3

# !/usr/bin/python3
# CSE354 Sp20; Assignment 1 Template v1
##################################################################

import sys

##################################################################
# 1. Tokenizer

import re  # python's regular expression package


def tokenize(sent):
    # input: a single sentence as a string.
    # output: a list of each "word" in the text
    # must use regular expressions
    pattern = r'(\b[A-z]+[-\']*[A-z]+\b|[A-Z\.]+|[@#]*[A-z]+|[,;:?!,~=`&(){}$\^\-#\*%\/])'
    tokens = re.findall(pattern, sent)
    # <FILL IN>
    return tokens


##################################################################
# 2. Pig Latinizer

def pigLatinizer(tokens):
    # input: tokens: a list of tokens,
    # output: plTokens: tokens after transforming to pig latin

    plTokens = []
    # <FILL IN>
    vowels = ['a', 'e', 'i', 'o', 'u']
    for token in tokens:
        if token[0].lower() in vowels:
            newToken = token + 'way'
            plTokens.append(newToken)
        elif not token[0].isalpha():
            plTokens.append(token)
        else:
            if not token.isalnum():
                plTokens.append(token)
                continue
            else:
                addedChars = ''
                newToken = token
                for char in token:
                    if char.lower() not in vowels:
                        addedChars += char
                        newToken = newToken.replace(char, '')
                    elif char.lower() in vowels:
                        break
                newToken += addedChars
                newToken += 'ay'
                plTokens.append(newToken)
    return plTokens


##################################################################
# 3. Feature Extractor

import numpy as np


def getFeaturesForTokens(tokens, wordToIndex):
    # input: tokens: a list of tokens,
    # wordToIndex: dict mapping 'word' to an index in the feature list.
    # output: list of lists (or np.array) of k feature values for the given target

    num_words = len(tokens)
    featuresPerTarget = list()  # holds arrays of feature per word

    for targetI in range(num_words):
        # <FILL IN>
        featureValues = np.zeros(len(wordToIndex))
        featurePrev = np.zeros(len(wordToIndex))
        featureNext = np.zeros(len(wordToIndex))

        featureValues.itemset(wordToIndex[tokens[targetI].lower()], 1)
        if targetI != 0 and targetI != (num_words-1):
            featurePrev.itemset(wordToIndex[tokens[targetI-1].lower()], 1)
            featureNext.itemset(wordToIndex[tokens[targetI+1].lower()], 1)

        featureValues = np.concatenate((featureValues,
                                        featurePrev,
                                        featureNext,
                                        [countVowels(tokens[targetI])],
                                        [countConsts(tokens[targetI])]))
        featuresPerTarget.append(featureValues)

    return featuresPerTarget  # a (num_words x k) matrix


def countVowels(word):
    count = 0
    vowels = ['a', 'e', 'i', 'o', 'u']
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


##################################################################
# 4. Adjective Classifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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
    print('\n  Best C for test data was ', bestFit.C, ' with an accuracy of %.4f' % bestAcc, '\n')
    return bestFit


##################################################################
##################################################################
## Main and provided complete methods
## Do not edit.
## If necessary, write your own main, but then make sure to replace
## and test with this before you submit.
##
## Note: Tests below will be a subset of those used to test your
##       code for grading.

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


# Main
if __name__ == '__main__':
    # Data for 1 and 2
    testSents = ['I am attending NLP class 2 days a week at S.B.U. this Spring.',
                 "I don't think data-driven computational linguistics is very tough.",
                 '@mybuddy and the drill begins again. #SemStart']

    # 1. Test Tokenizer:
    print("\n[ Tokenizer Test ]\n")
    tokenizedSents = []
    for s in testSents:
        tokenizedS = tokenize(s)
        print(s, tokenizedS, "\n")
        tokenizedSents.append(tokenizedS)

    # 2. Test Pig Latinizer:
    print("\n[ Pig Latin Test ]\n")
    for ts in tokenizedSents:
        print(ts, pigLatinizer(ts), "\n")

    # load data for 3 and 4 the adjective classifier data:
    taggedSents = getConllTags('daily547.conll')

    # 3. Test Feature Extraction:
    print("\n[ Feature Extraction Test ]\n")
    # first make word to index mapping:
    wordToIndex = set() # maps words to an index
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent) # splits [(w, t), (w, t)] into [w, w], [t, t]
            wordToIndex |= set([w.lower() for w in words]) # union of the words into the set
    print("  [Read ", len(taggedSents), " Sentences]")
    # turn set into dictionary: word: index
    wordToIndex = {w: i for i, w in enumerate(wordToIndex)}

    # Next, call Feature extraction per sentence
    sentXs = []
    sentYs = []
    print("  [Extracting Features]")
    for sent in taggedSents:
        if sent:
            words, tags = zip(*sent)
            sentXs.append(getFeaturesForTokens(words, wordToIndex))
            sentYs.append([1 if t == 'A' else 0 for t in tags])
    # test sentences
    print("\n", taggedSents[5], "\n", sentXs[5], "\n")
    print(taggedSents[192], "\n", sentXs[192], "\n")

    # 4. Test Classifier Model Building
    print("\n[ Classifier Test ]\n")
    # setup train/test:
    from sklearn.model_selection import train_test_split

    # flatten by word rather than sent:
    X = [j for i in sentXs for j in i]
    y = [j for i in sentYs for j in i]
    try:
        X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                            np.array(y),
                                                            test_size=0.20,
                                                            random_state=42)
    except ValueError:
        print("\nLooks like you haven't implemented feature extraction yet.")
        print("[Ending test early]")
        sys.exit(1)
    print("  [Broke into training/test. X_train is ", X_train.shape, "]")
    # Train the model.
    print("  [Training the model]")
    tagger = trainAdjectiveClassifier(X_train, y_train)
    print("  [Done]")

    # Test the tagger.
    from sklearn.metrics import classification_report

    # get predictions:
    y_pred = tagger.predict(X_test)
    # compute accuracy:
    leny = len(y_test)
    print("test n: ", leny)
    acc = np.sum([1 if (y_pred[i] == y_test[i]) else 0 for i in range(leny)]) / leny
    print("Accuracy: %.4f" % acc)
    #print(classification_report(y_test, y_pred, target_names=['not_adj', 'adjective']))