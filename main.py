import os
import json
import random
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Load intents file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

with open(INTENTS_PATH, 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Tokenize and lemmatize
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes
with open(os.path.join(BASE_DIR, 'words.pkl'), 'wb') as f:
    pickle.dump(words, f)

with open(os.path.join(BASE_DIR, 'classes.pkl'), 'wb') as f:
    pickle.dump(classes, f)

# Prepare training data
dataset = []
template = [0] * len(classes)

for document in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in words:
        bag.append(1 if word in pattern_words else 0)

    output_row = list(template)
    output_row[classes.index(document[1])] = 1
    dataset.append([bag, output_row])

random.shuffle(dataset)

train_x = np.array([item[0] for item in dataset])
train_y = np.array([item[1] for item in dataset])

# Build model
model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model
model.save(os.path.join(BASE_DIR, "chatbot_model.h5"))

# Optional: Save training history
with open(os.path.join(BASE_DIR, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

print("Training complete. Model and files saved.")
