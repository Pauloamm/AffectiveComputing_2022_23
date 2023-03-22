import os
from VoiceModelPrediction import GetVoicePrediction
import speech_recognition as sr
import pyttsx3
import main

# Initialize the recognizer


# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def GetVoiceInText(vectorizer, model):
    r = sr.Recognizer()


    while (1):
        try:

            # use the microphone as source for input.
            with sr.Microphone() as source2:

                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level
                r.adjust_for_ambient_noise(source2, duration=1)

                # listens for the user's input
                print("You may now talk")
                audio2 = r.listen(source2)

                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()
                print("Did you say ", MyText)
                SpeakText(MyText)


                #Store audio and get analysis from voice and text


                voiceRecordingPath = StoreAudioFile(audio2)
                GetVoicePrediction(voiceRecordingPath)
                main.getAnalysis(MyText)
                main.GetEmotionPrediction(MyText,vectorizer,model)


        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("unknown error occurred")

def StoreAudioFile(audioToStore):
    rootPath = os.getcwd()
    voiceRecordingFileName= 'audio_file.wav'
    voiceRecordingPath = os.path.join(rootPath,'VoiceRecordings',voiceRecordingFileName)

    with open(voiceRecordingPath, "wb") as file: file.write(audioToStore.get_wav_data())

    return voiceRecordingPath