import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import os
import glob
from sklearn.metrics import confusion_matrix

TESS = "DATASETS/TESS/"
RAVNESS = "DATASETS/RAVNESS/"
SAVEE = "DATASETS/SAVEE/"
CREMA = "DATASETS/CREMA-D/"


LOAD_TESS = True
LOAD_RAVNESS=True
LOAD_SAVEE = True
LOAD_CREMA = True

# SAVEE
dir_list = os.listdir(SAVEE)
# parse the filename to get the emotions
emotions = []
dir = []
for i in dir_list:
    if i[-8:-6] == '_a':
        emotions.append('angry')
    elif i[-8:-6] == '_d':
        emotions.append('disgust')
    elif i[-8:-6] == '_f':
        emotions.append('fear')
    elif i[-8:-6] == '_h':
        emotions.append('happy')
    elif i[-8:-6] == '_n':
        emotions.append('neutral')
    elif i[-8:-6] == 'sa':
        emotions.append('sad')
    elif i[-8:-6] == 'su':
        emotions.append('surprise')
    else:
        emotions.append('error')
    dir.append(SAVEE + i)

# Now check out the label count distribution
SAVEE_df = pd.DataFrame(emotions, columns=['labels'])
SAVEE_df['source'] = 'SAVEE'
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(dir, columns=['path'])], axis=1)
print(SAVEE_df.labels.value_counts())

# RAVNESS
dir_list = os.listdir(RAVNESS)
dir_list.sort()

emotions = []
gender = []
dir = []
for i in dir_list:
    filename = os.listdir(RAVNESS + i)
    for f in filename:
        part = f.split('.')[0].split('-')
        emotions.append(int(part[2]))
        temp = int(part[6])
        if temp % 2 == 0:
            temp = ""
        else:
            temp = ""
        gender.append(temp)
        dir.append(RAVNESS + i + '/' + f)

RAVNESS_df = pd.DataFrame(emotions)
RAVNESS_df = RAVNESS_df.replace(
    {1: 'neutral', 2: 'neutral', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'})
RAVNESS_df = pd.concat([pd.DataFrame(gender), RAVNESS_df], axis=1)
RAVNESS_df.columns = ['gender', 'emotion']
RAVNESS_df['labels'] = RAVNESS_df.gender  + RAVNESS_df.emotion
RAVNESS_df['source'] = 'RAVDESS'
RAVNESS_df = pd.concat([RAVNESS_df, pd.DataFrame(dir, columns=['path'])], axis=1)
RAVNESS_df = RAVNESS_df.drop(['gender', 'emotion'], axis=1)
RAVNESS_df.labels.value_counts()

# TESS

dir = []
emotions = []
dir_list = os.listdir(TESS)
dir_list.sort()

for i in dir_list:
    filename = os.listdir(TESS + i)
    for f in filename:
        if i == 'OAF_angry' or i == 'YAF_angry':
            emotions.append('angry')
        elif i == 'OAF_disgust' or i == 'YAF_disgust':
            emotions.append('disgust')
        elif i == 'OAF_Fear' or i == 'YAF_fear':
            emotions.append('fear')
        elif i == 'OAF_happy' or i == 'YAF_happy':
            emotions.append('happy')
        elif i == 'OAF_neutral' or i == 'YAF_neutral':
            emotions.append('neutral')
        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
            emotions.append('surprise')
        elif i == 'OAF_Sad' or i == 'YAF_sad':
            emotions.append('sad')
        else:
            emotions.append('Unknown')
        dir.append(TESS + i + "/" + f)

TESS_df = pd.DataFrame(emotions, columns=['labels'])
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df, pd.DataFrame(dir, columns=['path'])], axis=1)
TESS_df.labels.value_counts()

# CREMA

gender = []
emotions = []
dir = []
female = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 1025, 1028, 1029, 1030,
          1037, 1043, 1046, 1047, 1049,
          1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082, 1084,
          1089, 1091]
dir_list = os.listdir(CREMA)
dir_list.sort()

for i in dir_list:
    part = i.split('_')
    if int(part[0]) in female:
        temp = 'female'
    else:
        temp = 'male'
    gender.append(temp)
    if part[2] == 'SAD' and temp == 'male':
        emotions.append('sad')
    elif part[2] == 'ANG' and temp == 'male':
        emotions.append('angry')
    elif part[2] == 'DIS' and temp == 'male':
        emotions.append('disgust')
    elif part[2] == 'FEA' and temp == 'male':
        emotions.append('fear')
    elif part[2] == 'HAP' and temp == 'male':
        emotions.append('happy')
    elif part[2] == 'NEU' and temp == 'male':
        emotions.append('neutral')
    elif part[2] == 'SAD' and temp == 'female':
        emotions.append('sad')
    elif part[2] == 'ANG' and temp == 'female':
        emotions.append('angry')
    elif part[2] == 'DIS' and temp == 'female':
        emotions.append('disgust')
    elif part[2] == 'FEA' and temp == 'female':
        emotions.append('fear')
    elif part[2] == 'HAP' and temp == 'female':
        emotions.append('happy')
    elif part[2] == 'NEU' and temp == 'female':
        emotions.append('neutral')
    else:
        emotions.append('Unknown')
    dir.append(CREMA + i)

CREMA_df = pd.DataFrame(emotions, columns=['labels'])
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df, pd.DataFrame(dir, columns=['path'])], axis=1)
CREMA_df.labels.value_counts()

##JOIN DATASETS
df = pd.DataFrame()
if LOAD_SAVEE:
    df = pd.concat([SAVEE_df], axis=0)
if LOAD_TESS:
    df = pd.concat([df, TESS_df], axis=0)
if LOAD_RAVNESS:
    df = pd.concat([df, RAVNESS_df], axis=0)
if LOAD_CREMA:
    df = pd.concat([df, CREMA_df], axis = 0)

print(df.labels.value_counts())
df.head()
df.to_csv("DATASET_PATHS_MalenFemale.csv",index=False)