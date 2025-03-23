# 🩺 **Talking Healthcare Chatbot** 🤖

A **smart, NLP-based healthcare chatbot** that provides **symptom-based diagnosis** using deep learning. It integrates voice input/output, allowing users to interact through text or speech.

This project utilizes **Natural Language Processing (NLP)** for recognizing medical symptoms and conditions, with a **neural network** model built using **TensorFlow/Keras**. The chatbot is available both in **text-based** (CLI) and **voice-enabled** formats.

---

## 🧠 **Key Features:**

- **Symptom-based Diagnosis**: Provides possible conditions based on user symptoms.
- **Natural Language Processing (NLP)**: Uses tokenization, lemmatization, and classification models.
- **Deep Learning Model**: Neural network trained on medical conditions and symptoms.
- **Voice Integration**: Use speech-to-text for input and text-to-speech for responses.
- **CLI Interface**: Easy to interact with through a terminal/command line.
- **Extendable**: You can add more symptoms and conditions to the chatbot.

---

'''
## 📂 **Project Structure:**
talking-healthcare-bot/ ├── intents.json # Medical conditions and symptom patterns data ├── main.py # Training script for building the model ├── chatbot_response.py # Predicts the response based on symptoms ├── cli_chatbot.py # Text-based chatbot interface (CLI) ├── talking_bot.py # Voice-enabled chatbot interface (speech input/output) ├── classes.pkl # Auto-generated: list of unique tags (conditions) ├── words.pkl # Auto-generated: vocabulary list ├── chatbot_model.h5 # Auto-generated: trained model file ├── training_history.pkl # Auto-generated: training history (optional) └── README.md # Project documentation (this file)

'''

## 💻 **Setup Instructions:**

### **1. Clone the Repository**
git clone https://github.com/PrincePandit16/talking-healthcare-bot.git
cd talking-healthcare-bot


### **2. Install Required Libraries**
pip install nltk tensorflow speechrecognition pyttsx3 pyaudio


### **3. Train the Model**
python main.py


### **4. Run the Text Chatbot**
python cli_chatbot.py


### **5. Run the Talking Bot (Voice Enabled)**
python talking_bot.py


## 🤖 **How It Works:**
1. Preprocess user sentences (tokenization + lemmatization)
2. Convert to Bag-of-Words vector
3. Predict intent using trained neural network
4. Respond with relevant symptom/condition guidance


## 📡 **Dataset Info:**
All intent patterns and responses are defined in intents.json. You can expand the bot by adding more patterns or medical tags easily.

## 💡 **Future Improvements:**
GUI-based chatbot (Tkinter or Web UI)
API integration with real diagnosis database
Multi-language support

## 📬 **Contact**
Developer: Prince Pandit
