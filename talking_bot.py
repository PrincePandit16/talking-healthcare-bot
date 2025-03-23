import speech_recognition as sr
import pyttsx3
from chatbot_response import predict_class, get_response, intents

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    print(f"Bot: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            print(f"You: {query}")
            return query
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that.")
            return ""
        except sr.RequestError:
            speak("Speech service is unavailable.")
            return ""

# Main loop
speak("Hi, I am your talking healthcare assistant. You can ask about your symptoms. Say 'stop' to exit.")
while True:
    query = listen()
    if query.lower() in ["stop", "quit", "exit"]:
        speak("Take care! Goodbye!")
        break
    if query:
        intents_list = predict_class(query)
        response = get_response(intents_list, intents)
        speak(response)
