from sklearnex import patch_sklearn # <3 <3 <3

import os
import random
import pandas as pd
import VoiceToText

##Sentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


#Emotions

#NLP
from sklearn.feature_extraction.text import CountVectorizer


#Utils
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



#plot
import matplotlib.pyplot as plt
import seaborn as sns



def main():

    dataset = GetDataset()
    datasetSize = dataset.shape[0]

    # Emotion
    vectorizer, model= ModelsTraining(dataset)

    VoiceToText.GetVoiceInText(vectorizer, model)



    #randomSentenceToAnalyze= dataset[0][random.randint(0, datasetSize)]
    #randomSentenceToAnalyze= "I am so sad now"

    #Polarity/Subjectivity
    #getAnalysis(randomSentenceToAnalyze)



    #blob = TextBlob('His naime ise John')
    #print(blob.correct())

def GetDataset():

    rootPath = os.getcwd()
    datasetsPath = os.path.join(rootPath, 'DATASETS')
    dataset1Path = os.path.join(datasetsPath, 'DATASET_1')
    dataset2Path = os.path.join(datasetsPath, 'DATASET_2')

    dataset1 = pd.read_csv(os.path.join(dataset1Path, 'dataset1.csv'), delimiter=';', header=None)

    # with open(os.path.join(dataset2Path,'merged_training.pkl'), 'rb') as f:
    #    dataset2 = pickle.load(f)
    dataset2 = pd.read_pickle(os.path.join(dataset2Path, 'merged_training.pkl'))  # DATASET NOT USED, BAD

    return dataset1

def getAnalysis(text):

    print("Sentence to Analyze: " + text)

    TextBlobAnalysis(text)# Polarity/Subjectivity
    VADERAnalysis(text) #Polarity

def TextBlobAnalysis(text):
    print("\nUSING TEXTBLOB: ")

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
    print("\nUSING VADER: ")

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

    patch_sklearn()
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    #import xgboost as xgb
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC, SVC


    # Splits info
    columnNames = datasetForTraining.columns

    xValuesRaw = datasetForTraining[columnNames[:-1]]
    xValues = []
    xValuesTokenized = []

    for index, row in xValuesRaw.iterrows():
        value = row.iloc[0]
        xValues.append(value)

    yValuesRaw = datasetForTraining[columnNames[-1]]
    yValues=[]

    for _, value in yValuesRaw.items():
        yValues.append(value)

    vectorizer = CountVectorizer(min_df=2,ngram_range=(2,2)) #eliminates low occurrence words(2 or less sentences)
    vectorizer.fit_transform(xValues) # fits vocabulary
    tokenizedX = vectorizer.transform(xValues)


    classificationModelsDictionary = \
        {
            1: (LinearSVC(), 'LinearSVC Model'),
            2: (SVC(), 'SVC Model'),
            3: (MultinomialNB(), 'MultinomialNB Model'),
            4: (BernoulliNB(), 'BernoulliNB Model'),
            #5: (xgb.XGBClassifier(objective="multi:softprob", random_state=42), 'XGBoost'),
            5: (KNeighborsClassifier(),'KNeighborsClassifier Model')

        }

    percentageForTraining = 0.7

    xValuesToTrain, xValuesToTest, yValuesToTrain, yValuesToTest = train_test_split(tokenizedX, yValues, test_size=(
            1 - percentageForTraining), random_state=1)

    bestModel = 0
    bestAccuracy = 0
    for counter in range(1, len(classificationModelsDictionary) + 1):

        modelName = classificationModelsDictionary[counter][1]

        model = classificationModelsDictionary[counter][0]
        model.fit(xValuesToTrain, yValuesToTrain)

        yValuesPrediction = model.predict(xValuesToTest)

        accuracy = accuracy_score(yValuesToTest,yValuesPrediction)
        print("\nClassification Report of {}".format(modelName))
        print("Accuracy: " +str(accuracy) + "\n\n")

        if accuracy >= bestAccuracy:
            bestAccuracy=accuracy
            bestModel=model

        classNames = model.classes_
        cm = confusion_matrix(yValuesToTest, yValuesPrediction, labels=classNames)
        PlotConfusionMatrix(classNames,cm,modelName)

    return vectorizer,bestModel


def GetEmotionPrediction(sentenceToAnalyze, vectorizer,bestModel):
    prediction = bestModel.predict(vectorizer.transform([sentenceToAnalyze]))
    print("Emotion predicted by best model: {}".format(prediction))

#https://stackoverflow.com/questions/65618137/confusion-matrix-for-multiple-classes-in-python
def PlotConfusionMatrix(classNames,confusionMatrix,modelName):

    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(16, 14))
    ax = plt.subplot()
    sns.heatmap(confusionMatrix, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(classNames, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(classNames, fontsize=10)
    plt.yticks(rotation=0)

    plt.title(modelName, fontsize=20)
    #plt.savefig(modelName + '_'+'CM.png')
    #plt.show()




if __name__ == '__main__':
    main()

