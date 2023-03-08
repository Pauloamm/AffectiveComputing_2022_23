
import os
import pickle

import pandas as pd

##Sentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#Emotions

#NLP
from nltk.tokenize import word_tokenize

#Utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Classification Models
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


def main():
    rootPath = os.getcwd()
    datasetsPath = os.path.join(rootPath,'DATASETS')
    dataset1Path = os.path.join(datasetsPath,'DATASET_1')
    dataset2Path = os.path.join(datasetsPath,'DATASET_2')



    dataset1 = pd.read_csv(os.path.join(dataset1Path,'dataset1.csv'),delimiter=';',header=None)

    #with open(os.path.join(dataset2Path,'merged_training.pkl'), 'rb') as f:
    #    dataset2 = pickle.load(f)
    dataset2 = pd.read_pickle(os.path.join(dataset2Path,'merged_training.pkl'))

    #teste = dataset1[0][10]
    teste= dataset1[0][0]


    #getAnalysis(teste)

    ModelsTraining(dataset1)

    #blob = TextBlob('His naime ise John')
    #print(blob.correct())

def getAnalysis(text):

    print("Sentence to Analyze: " + text)

    TextBlobAnalysis(text)# Polarity/Subjectivity
    VADERAnalysis(text) #Polarity

def TextBlobAnalysis(text):
    print("USING TEXTBLOB: ")

    tBlob = TextBlob(text)
    pol = tBlob.polarity  # Between -1,1 (negative,positive)

    if pol < 0:
        print('Polarity: Negative')
    elif pol > 0:
        print('Polarity: Positive')
    else:
        print('Polarity: Neutral')

    print("Polarity Value = " + str(pol))
    sub = tBlob.subjectivity

    if sub < 0.5:
        print('Subjectivity: Fact')
    else:
        print('Subjectivity: Opinion')
    print("Subjectivity Value = " + str(sub))

def VADERAnalysis(text):
    print("USING VADER: ")

    sentimentAnalyzer = SentimentIntensityAnalyzer()
    sentiment_dict = sentimentAnalyzer.polarity_scores(text)

    print(sentiment_dict['neg'] * 100, "% Negative")
    print(sentiment_dict['neu'] * 100, "% Neutral")
    print(sentiment_dict['pos'] * 100, "% Positive")

    print("Sentence Overall Rated As", end=" ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        print("Positive")

    elif sentiment_dict['compound'] <= - 0.05:
        print("Negative")

    else:
        print("Neutral")

def ModelsTraining(datasetForTraining):




    # Splits info
    columnNames = datasetForTraining.columns

    xValues = datasetForTraining[columnNames[:-1]]

    for x in xValues:
        tokenizedX = word_tokenize(x)
        print(tokenizedX)

    yValues = datasetForTraining[columnNames[-1]]



    classificationModelsDictionary = \
        {
            1: (LinearSVC(), 'LinearSVC Model'),
            2: (KNeighborsClassifier(), 'KNeighborsClassifier Model'),
            3: (DecisionTreeClassifier(), 'DecisionTreeClassifier Model'),
            4: (RandomForestClassifier(), 'RandomForestClassifier Model'),
            5: (AdaBoostClassifier(), 'AdaBoostClassifier Model'),
            6: (MLPClassifier(), 'MLPClassifier Model'),

        }

    percentageForTraining = 0.7

    xValuesToTrain, xValuesToTest, yValuesToTrain, yValuesToTest = train_test_split(xValues, yValues, test_size=(
            1 - percentageForTraining), random_state=1)

    numberOfLines = 3
    numberOfColumns = 2

    #fig, axs = plt.subplots(numberOfLines, numberOfColumns)
    #fig.suptitle("Quality According to {} using:".format(columnNames[bestColumnToTestDependency]))

    for counter in range(1, len(classificationModelsDictionary) + 1):
        model = classificationModelsDictionary[counter][0]

        model.fit(xValuesToTrain, yValuesToTrain)

        yValuesPrediction = model.predict(xValuesToTest)

        print("\nClassification Report of {}".format(classificationModelsDictionary[counter][1]))
        #print(classification_report(yValuesToTest, yValuesPrediction, zero_division=1))
        print("Accuracy: " +str(accuracy_score(yValuesToTest,yValuesPrediction)) + "\n\n")

if __name__ == '__main__':
    main()

