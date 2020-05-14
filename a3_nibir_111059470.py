import numpy as np

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error as MAE

import pandas as pd
import scipy.stats as ss
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer

import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Bidirectional
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.initializers import Constant

import sys


class Product:
    def __init__(self, productID, rating, userID, reviewText):
        self.productID = productID
        self.rating = rating
        self.userID = userID
        self.reviewText = reviewText


def indexFromDict(word, d):
    return list(d.keys()).index(word)


if __name__ == '__main__':
    # Read command line arguments
    args = sys.argv
    trainData = args[1]
    trialData = args[2]

    # Get data from csv file
    # Train data
    #productDataTrain = pd.read_csv(trainData)
    productDataTrain = pd.read_csv(trainData)
    productIDsTrain = productDataTrain['id']
    productRatingsTrain = productDataTrain['rating']
    productUserIdsTrain = productDataTrain['user_id']
    productReviewTextsTrain = productDataTrain['reviewText']

    # Trial data
    #productDataTrial = pd.read_csv(trialData)
    productDataTrial = pd.read_csv(trialData)
    productIDsTrial = productDataTrial['id']
    productRatingsTrial = productDataTrial['rating']
    productUserIdsTrial = productDataTrial['user_id']
    productReviewTextsTrial = productDataTrial['reviewText']


    # Create dict of product objects from train set
    productTrainDict = {}
    reviewTextsTokenized = []
    ratingsTrain = []
    tknzer = TweetTokenizer()
    for i in range(len(productIDsTrain)):
        productReviewTokenized = tknzer.tokenize(str(productReviewTextsTrain[i]))
        reviewTextsTokenized.append(productReviewTokenized)
        newProduct = Product(productIDsTrain[i],
                             productRatingsTrain[i],
                             productUserIdsTrain[i],
                             productReviewTokenized)
        ratingsTrain.append(productRatingsTrain[i])
        productTrainDict[productIDsTrain[i]] = newProduct

    #print(len(productTrainDict))

    # Create word2vec embeddings for words in train set reviews
    word2vecModel = Word2Vec(reviewTextsTokenized, size=128, min_count=5)
    #print(word2vecModel.wv.vocab)

    meanTrainVec = []
    for product in productTrainDict.values():
        meanSentVec = []
        for word in product.reviewText:
            if word in word2vecModel.wv.vocab.keys():
                wordVec = word2vecModel.wv.get_vector(word)
                meanSentVec.append(wordVec)
            else:
                wordVec = np.zeros(128)
                meanSentVec.append(wordVec)
        meanSentVec = np.mean(meanSentVec, axis=0)
        meanSentVec = meanSentVec - np.mean(meanSentVec)
        meanTrainVec.append(meanSentVec)


    # Create linear regression classifier and fit to train data
    meanTrainVec = np.array(meanTrainVec)
    meanTrainVec = meanTrainVec.reshape((len(meanTrainVec), 128))
    ratingsTrain = np.array(ratingsTrain)
    ratingsTrain = ratingsTrain.reshape(len(ratingsTrain), 1)
    #print(meanTrainVec.shape)
    ridgeClassifier = Ridge(alpha=0.01)
    ridgeClassifier.fit(meanTrainVec, ratingsTrain)

    # Create product objects from trial set
    productTrialDict = {}
    reviewTextsTokenizedTrial = []
    ratingsTrial = []
    for i in range(len(productIDsTrial)):
        productReviewTokenized = tknzer.tokenize(str(productReviewTextsTrain[i]))
        reviewTextsTokenizedTrial.append(productReviewTokenized)
        newProduct = Product(productIDsTrain[i],
                             productRatingsTrain[i],
                             productUserIdsTrain[i],
                             productReviewTokenized)
        ratingsTrial.append(productRatingsTrain[i])
        productTrialDict[productIDsTrial[i]] = newProduct

    meanTrialVec = []
    for review in reviewTextsTokenizedTrial:
        meanSentVec = []
        for word in review:
            if word in word2vecModel.wv.vocab.keys():
                wordVec = word2vecModel.wv.get_vector(word)
                meanSentVec.append(wordVec)
            else:
                wordVec = np.zeros(128)
                meanSentVec.append(wordVec)
        meanSentVec = np.mean(meanSentVec, axis=0)
        meanSentVec -= np.mean(meanSentVec)
        meanTrialVec.append(meanSentVec)

    # Create trial data for classifier
    meanTrialVec = np.array(meanTrialVec)
    meanTrialVec = meanTrialVec.reshape((len(meanTrialVec), 128))
    ratingsTrial = np.array(ratingsTrial, dtype=float)
    ratingsTrial = ratingsTrial.reshape(len(ratingsTrial), 1)

    predictedRatings = ridgeClassifier.predict(meanTrialVec)
    predictedRatings = np.array(predictedRatings, dtype=float)

    #print(ratingsTrial)
    #print(predictedRatings)

    meanAbsoluteError = MAE(ratingsTrial, predictedRatings)
    pearsonCoeff = ss.pearsonr(ratingsTrial.flatten(), predictedRatings.flatten())

    print('\nStage One Checkpoint: \n')
    print('MAE: ', meanAbsoluteError)
    print('Pearson product-moment correlation: ', pearsonCoeff[0], '\n')

    testCases = [548, 4258, 4766, 5800]
    for testCase in testCases:
        if testCase >= ratingsTrial.shape[0]:
            print(testCase, ' does not exist in trial data')
        else:
            print('Predicted value of id ', testCase, ': ', predictedRatings[testCase])
            print('Actual value of id ', testCase, ': ', ratingsTrial[testCase], '\n')

    ##################################################################
    # 2. User-Factor Adaption

    # Create dict of users to the reviews for train data
    userFactorTrainDict = {}
    for i, product in zip(range(len(productTrainDict)), productTrainDict.values()):
        if product.userID not in userFactorTrainDict.keys():
            userFactorTrainDict[product.userID] = [meanTrainVec[i]]
        else:
            avgEmbedding = meanTrainVec[i]
            userFactorTrainDict[product.userID].append(avgEmbedding)

    #print(len(userFactorTrainDict))

    userFactorTrainMeanVec = []
    for userId, userFeatures in userFactorTrainDict.items():
        meanVec = np.mean(userFeatures, axis=0)
        meanVec -= np.mean(meanVec)
        userFactorTrainDict[userId] = meanVec
        userFactorTrainMeanVec.append(meanVec)

    userFactorTrainMeanVec = np.array(userFactorTrainMeanVec)
    #print(userFactorTrainMeanVec.shape)

    # Reduce dimensions using PCA
    pca = PCA(n_components=3, svd_solver='randomized')
    pca.fit(userFactorTrainMeanVec)

    # Create dict of users to the reviews for trial data
    userFactorTrialDict = {}
    for i, product in zip(range(len(productTrialDict)), productTrialDict.values()):
        if product.userID not in userFactorTrialDict.keys():
            userFactorTrialDict[product.userID] = [meanTrialVec[i]]
        else:
            avgEmbedding = meanTrialVec[i]
            userFactorTrialDict[product.userID].append(avgEmbedding)

    #print(len(userFactorTrialDict))

    userFactorTrialMeanVec = []
    for userId, userFeatures in userFactorTrialDict.items():
        meanVec = np.mean(userFeatures, axis=0)
        meanVec -= np.mean(meanVec)
        userFactorTrialDict[userId] = meanVec
        userFactorTrialMeanVec.append(meanVec)

    userFactorTrialMeanVec = np.array(userFactorTrialMeanVec)
    #print(userFactorTrialMeanVec.shape)

    reducedEmbeddings = pca.transform(userFactorTrainMeanVec)
    transformMatrix = np.transpose(pca.components_)
    #print(reducedEmbeddings)
    #print(transformMatrix.shape)

    for i, userID in zip(range(len(reducedEmbeddings)), userFactorTrainDict.keys()):
        userFactorTrainDict[userID] = reducedEmbeddings[i]

    finalTrainReducedEmb = []
    for emb, product in zip(meanTrainVec, productTrainDict.values()):
        newEmb = []
        userFactorVec = userFactorTrainDict[product.userID]
        for i in range(len(userFactorVec)):
            newEmb.extend(emb*userFactorVec[i])
        finalTrainReducedEmb.append(newEmb)

    # Apply transformation matrix on trial data
    reducedEmbeddingsTrial = np.dot(userFactorTrialMeanVec, transformMatrix)
    #print(reducedEmbeddingsTrial.shape)

    for i, userID in zip(range(len(reducedEmbeddingsTrial)), userFactorTrialDict.keys()):
        userFactorTrialDict[userID] = reducedEmbeddingsTrial[i]

    finalTrialReducedEmb = []
    for emb, product in zip(meanTrialVec, productTrialDict.values()):
        newEmb = []
        userFactorVec = userFactorTrialDict[product.userID]
        for i in range(len(userFactorVec)):
            newEmb.extend(emb*userFactorVec[i])
        finalTrialReducedEmb.append(newEmb)

    finalTrainReducedEmb = np.array(finalTrainReducedEmb)
    finalTrialReducedEmb = np.array(finalTrialReducedEmb)
    #print(finalTrainReducedEmb.shape)

    # Use ridge regression on user factor features
    ridgeClassifier = Ridge(alpha=0.0001)
    ridgeClassifier.fit(finalTrainReducedEmb, ratingsTrain)

    predictedRatings = ridgeClassifier.predict(finalTrialReducedEmb)

    meanAbsoluteError = MAE(ratingsTrial, predictedRatings)
    pearsonCoeff = ss.pearsonr(ratingsTrial.flatten(), predictedRatings.flatten())

    print('\nStage Two Checkpoint: \n')
    print('MAE: ', meanAbsoluteError)
    print('Pearson product-moment correlation:, ', pearsonCoeff[0], '\n')

    testCases = [548, 4258, 4766, 5800]
    for testCase in testCases:
        if testCase >= ratingsTrial.shape[0]:
            print(testCase, ' does not exist in trial data')
        else:
            print('Predicted value of id ', testCase, ': ', predictedRatings[testCase])
            print('Actual value of id ', testCase, ': ', ratingsTrial[testCase], '\n')

    ##################################################################
    # 3. Deep Learning

    productDataTrain = pd.read_csv(trainData)
    ratingsTrain = productDataTrain['rating']
    reviewsTrain = productDataTrain['reviewText']

    productDataTrial = pd.read_csv(trialData)
    ratingsTest = productDataTrial['rating']
    reviewsTest = productDataTrial['reviewText']

    X_train = []
    for i in range(len(reviewsTrain)):
        X_train.append(str(reviewsTrain[i]))

    y_train = []
    for i in range(len(ratingsTrain)):
        y_train.append(int(ratingsTrain[i]))

    X_test = []
    for i in range(len(reviewsTest)):
        X_test.append(str(reviewsTest[i]))

    y_test = []
    for i in range(len(ratingsTest)):
        y_test.append(int(ratingsTest[i]))

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    tokenizer = Tokenizer()
    totalReviews = X_train + X_test
    tokenizer.fit_on_texts(totalReviews)

    tweetTokenizer = TweetTokenizer()
    totalReviewsTokenized = []
    for review in totalReviews:
        review = review.lower()
        tokenizedRev = tweetTokenizer.tokenize(review)
        totalReviewsTokenized.append(tokenizedRev)

    embDim = 128

    word2vecModel = Word2Vec(totalReviewsTokenized, size=embDim, min_count=1)

    maxLength = max([len(s.split()) for s in totalReviews])

    vocabSize = len(tokenizer.word_index) + 1

    embeddingWeights = np.zeros((vocabSize, embDim))

    for word, i in tokenizer.word_index.items():
        if i > vocabSize:
            continue
        if word in word2vecModel.wv.vocab.keys():
            embeddingWeights[i] = word2vecModel.wv.get_vector(word)

    XTrainTokens = tokenizer.texts_to_sequences(X_train)
    XTrainPad = pad_sequences(XTrainTokens, maxlen=maxLength, padding='post')
    XTestTokens = tokenizer.texts_to_sequences(X_test)
    XTestPad = pad_sequences(XTestTokens, maxlen=maxLength, padding='post')

    biGRU = Sequential()
    biGRU.add(Embedding(vocabSize, embDim, embeddings_initializer=Constant(embeddingWeights),
                        input_length=maxLength, mask_zero=True))
    biGRU.add(Bidirectional(GRU(units=20, dropout=0.3)))
    biGRU.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    lossFunction = tf.keras.losses.MeanSquaredError()
    biGRU.compile(optimizer=optimizer, loss=lossFunction)

    print('\nTraining Deep Learning Model\n')
    biGRU.fit(XTrainPad, y_train, batch_size=256, epochs=20)

    # model.save('my_model.h5')

    preds = biGRU.predict(XTestPad)

    meanAbsoluteError = MAE(y_test, preds)
    pearsonCoeff = ss.pearsonr(y_test.flatten(), preds.flatten())

    #print(meanAbsoluteError, pearsonCoeff[0])

    print('\n\nDeep Learning Accuracy: \n')
    print('MAE: ', meanAbsoluteError)
    print('Pearson product-moment correlation:, ', pearsonCoeff[0], '\n')
