import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\\Users\\21690\\PycharmProjects\\IAchatbot\\intent.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    highest_probability = 0
    selected_intent = None
    list_of_intents = intents_json['intents']

    for intent_data in intents_list:
        if float(intent_data['probability']) > highest_probability:
            highest_probability = float(intent_data['probability'])
            selected_intent = intent_data['intent']

    if selected_intent:
        for intent in list_of_intents:
            if intent['tag'] == selected_intent:
                result = random.choice(intent['responses'])
                break
        return result
    else:
        return "I'm sorry, I didn't understand that."


print("GO! Bot is running!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)


# python -u '"C:\\Users\\21690\\PycharmProjects\\IAchatbot\\"chatbot.py'
#cntr shift n = new.py
