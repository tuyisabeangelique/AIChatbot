# importing the required modules. 
import random 
import json 
import pickle 
import numpy as np 
import nltk 
from nltk.stem import WordNetLemmatizer 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Note: You may need to download these NLTK packages manually the first time.
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer() 

# reading the json.intents file 
intents = json.loads(open("intents.json").read()) 

# creating empty lists to store data 
words = [] 
classes = [] 
documents = [] 
letters_to_ignore = ["?", "!", ".", ","] 

for intent in intents['intents']: 
	for pattern in intent['patterns']: 
		# separating words from patterns 
		word_list = nltk.word_tokenize(pattern) 
		words.extend(word_list) # and adding them to words list 
		
		# associating patterns with respective tags 
		documents.append((word_list, intent['tag'])) 

		# appending the tags to the class list 
		if intent['tag'] not in classes: 
			classes.append(intent['tag']) 

# storing the root words or lemma 
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in letters_to_ignore] 
words = sorted(list(set(words))) 

# saving the words and classes list to binary files 
pickle.dump(words, open('words.pkl', 'wb')) 
pickle.dump(classes, open('classes.pkl', 'wb')) 

# we need numerical values of the 
# words because a neural network 
# needs numerical values to work with 
training_data = [] 
output_empty = [0] * len(classes) 

for document in documents: 
    bag = [] 
    word_patterns = document[0] 
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] 
    for word in words: 
        bag.append(1) if word in word_patterns else bag.append(0) 
    
    # making a copy of the output_empty 
    output_row = list(output_empty) 
    output_row[classes.index(document[1])] = 1
    training_data.append([bag, output_row]) 
    
random.shuffle(training_data) 
training_data = np.array(training_data, dtype=object)

# splitting the data 
train_x = list(training_data[:, 0]) 
train_y = list(training_data[:, 1])

# creating a Sequential machine learning model 
model = Sequential() 
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(len(train_y[0]), activation='softmax')) 
  
# compiling the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) 
  
model.save('my_model.keras')
  
print("Model created successfully!") 