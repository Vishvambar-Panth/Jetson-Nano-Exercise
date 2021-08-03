# This program uses the Python Text-to-speech module pyttsx3 to speak the sentence
import os
import pyttsx3 #pyttsx3 is a text-to-speech conversion library in Python
engine=pyttsx3.init() #Constructs a new TTS engine instance
engine.setProperty('rate',150) # Adds a property value to set to the event queue. Valid names and values include: voice: String ID of the voice rate: Integer speech rate in words per minute. volume: Floating point volume of speech in the range [0.0, 1.0]
engine.setProperty('voice','english+m1') #Speak in english
'''
voices = engine.getProperty('voices') # To know about diffrent available languages
for voice in voices:
    # to get the info. about various voices in our PC  
    print("Voice:") 
    print("ID: %s" %voice.id) 
    print("Name: %s" %voice.name) 
    print("Age: %s" %voice.age) 
    print("Gender: %s" %voice.gender) 
    print("Languages Known: %s" %voice.languages) 
'''
text='Welcome Rajapriyan to Artificial Intelligence World'
engine.say(text) # Utter the text
engine.runAndWait() # Wait until the current text is completed