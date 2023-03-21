import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from tensorflow.keras import Model
import tensorflow as tf
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tools import plot_history, data_set, print_confusion_matrix
from keras.utils import np_utils, to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
# data = pd.read_csv("(1)DATASET_PATHS_MalenFemale.csv")
data = pd.read_csv("(1)DATASET_PATHS_teste.csv")
print(data.head())

df = pd.DataFrame(columns=['features'])

counter = 0

for index, path in tqdm(enumerate(data.path)):
    X, sample_rate = librosa.load(path
                                  , res_type='kaiser_fast'
                                  , duration=2.5
                                  , sr=44100
                                  , offset=0.5
                                  )
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X,
                                         sr=sample_rate,
                                         n_mfcc=13),
                    axis=0)
    df.loc[counter] = [mfccs]
    counter = counter + 1

df = pd.concat([data, pd.DataFrame(df['features'].values.tolist())], axis=1)
# replace NA with 0
df = df.fillna(0)
print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['path', 'labels', 'source'], axis=1)
                                                    , df.labels
                                                    , test_size=0.25
                                                    , shuffle=True
                                                    , random_state=42
                                                    )
# normalization

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

print(X_train.shape)
print(lb.classes_)

# 1D CNN needs reshape
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
print(X_test.shape)

model = keras.models.load_model("1D_CNN_ALLDATA")
model.summary()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

predictions = model.predict(X_test)

predictions = predictions.argmax(axis=1)

predictions = predictions.astype(int).flatten()
predictions = (lb.inverse_transform((predictions)))
predictions = pd.DataFrame({'predictedvalues': predictions})

# labels
actual = y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform((actual)))
actual = pd.DataFrame({'actualvalues': actual})

predictions_df = actual.join(predictions)

# Write out the predictions to disk
predictions_df.to_csv('Predictions.csv', index=False)
predictions_df.groupby('predictedvalues').count()

# confusion matrix
