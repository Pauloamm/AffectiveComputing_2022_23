import numpy
import tensorflow.keras as keras
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

from model import build_model


def GetVoicePrediction(audioFilePath):
    model = keras.models.load_model("1D_CNN_ALLDATA")
    print(model.summary())

    # audioFilePath = os.path.join(os.getcwd(),'VoiceRecordings','audio_file.wav')

    X, sample_rate = librosa.load(audioFilePath
                                  , res_type='kaiser_fast'
                                  , duration=2.5
                                  , sr=44100
                                  , offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),
                    axis=0)

    xValue = [mfccs]


    inputShape = model.layers[0].input_shape
    print(inputShape)




    # normalization
    #mean = np.mean(xValue, axis=0)
    #std = np.std(xValue, axis=0)
    #xValue = (xValue - mean) / std
    xValue = np.array(xValue)



    # in case audio small fill with 0s the rest until 2.5sec
    if inputShape[1] > xValue.shape[1]:

        temp = numpy.zeros((1,inputShape[1]))
        temp[0,:xValue.shape[1]] = xValue
        xValue = temp



    xValue = np.expand_dims(xValue, axis=2)


    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes.npy', allow_pickle=True)

    prediction = model.predict(xValue)
    prediction = prediction.argmax(axis=1)
    prediction = encoder.inverse_transform((prediction))

    print(prediction)



