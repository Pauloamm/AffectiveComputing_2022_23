import pandas as pd
import librosa
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint
import pickle
from keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from model import build_model
import os
from tqdm import tqdm
import seaborn as sns


data = pd.read_csv("(1)DATASET_PATHS_MalenFemale.csv")
# data = pd.read_csv("DATASET_PATHS_SAVEE.csv")
print(data.head())

df = pd.DataFrame(columns=['features'])

counter = 0
checkpoint = ModelCheckpoint(filepath='1D_CNN_ALLDATA_24_mfccs',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

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
                                         n_mfcc=24),
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

# Pickel the lb object for future use
filename = 'labels'
outfile = open(filename, 'wb')
pickle.dump(lb, outfile)
outfile.close()

# 1D CNN needs reshape
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_train.shape

input_shape = (X_train.shape[1], 1)
model = build_model(input_shape)
print(model.summary())

model_name = "1D_CNN_ALLDATA_24_mfccs"
opt = optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model_history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=checkpoint)
model.save(model_name)

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




