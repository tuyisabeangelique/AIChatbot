import nltk 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
import random 
import json 
import pickle 
import numpy as np 
from keras.models import load_model 
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer() 
  
# loading the files we made previously 
intents = json.loads(open("intents.json").read()) 
words = pickle.load(open('words.pkl', 'rb')) 
classes = pickle.load(open('classes.pkl', 'rb')) 
model = load_model('chatbotmodel.h5') 


def clean_up_sentences(sentence): 
    sentence_words = nltk.word_tokenize(sentence) 
    sentence_words = [lemmatizer.lemmatize(word)  
                      for word in sentence_words] 
    return sentence_words 
  

def bagw(sentence): 
    
    # separate out words from the input sentence 
    sentence_words = clean_up_sentences(sentence) 
    bag = [0]*len(words) 
    for w in sentence_words: 
        for i, word in enumerate(words): 
  
            # check whether the word 
            # is present in the input as well 
            if word == w: 
  
                # as the list of words 
                # created earlier. 
                bag[i] = 1
  
    # return a numpy array 
    return np.array(bag)

def predict_class(sentence): 
    bow = bagw(sentence) 
    res = model.predict(np.array([bow]))[0] 
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res)  
               if r > ERROR_THRESHOLD] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_list = [] 
    for r in results: 
        return_list.append({'intent': classes[r[0]], 
                            'probability': str(r[1])}) 
        return return_list 
      
def get_response(intents_list, intents_json): 
    tag = intents_list[0]['intent'] 
    list_of_intents = intents_json['intents'] 
    result = "" 
    for i in list_of_intents: 
        if i['tag'] == tag: 
            
              # prints a random response 
            result = random.choice(i['responses'])   
            break
    return result 
  
print("Chatbot is up!")


while True: 
    message = input("") 
    ints = predict_class(message) 
    res = get_response(ints, intents) 
    print(res)

# Source: https://www.techwithtim.net/tutorials/ai-chatbot/part-1

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random

# Loading our JSON data. It is stored in the variable data
import json
with open('intents.json') as file:
    data = json.load(file)
    
# Taking out the data we want from our JSON file. We need all of the patterns and which class/tag they belong to.
words = []
labels = []
docs_x = []
docs_y = []

# Looping through our JSON data and extract the data we want. For each pattern we will turn it into a list of words using nltk.word_tokenizer, rather than having them as strings. We will then add each pattern into our docs_x list and its associated tag into the docs_y list.

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
        

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

# Converting training data and output to numpy arrays.
training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# Pass the model all of our training data. 
#  The number of epochs, n_epoch, we set is the amount of times that the model will see the same information while training.
try: 
  model.load("model.tflearn")
except:
  model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
  model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


# Here, we actually use the model! Ideally we want to generate a response to any sentence the user types in. To do this we need to remember that our model does not take string input, it takes a bag of words. We also need to realize that our model does not spit out sentences, it generates a list of probabilities for all of our classes. This makes the process to generate a response look like the following: – Get some input from the user – Convert it to a bag of words – Get a prediction from the model – Find the most probable class – Pick a response from that class

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()
