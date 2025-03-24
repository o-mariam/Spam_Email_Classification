#from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


dataset = pd.read_csv('spam_or_not_spam.csv')


dataset.email = dataset.email.astype(str)

emails = dataset["email"].values
labels = dataset["label"].values

print(len(emails))

#δημιουργία tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(emails)
vocab_size = len(tokenizer.word_index) + 1

#κάνουμε integer encode στα email που έχουμε
encoded_emails = tokenizer.texts_to_sequences(emails)
print(vocab_size)

#κανουμε padding στα emails
max_length = 20
padded_emails = pad_sequences(encoded_emails, maxlen=max_length, padding='post')
print(len(padded_emails))


#κάνουμε load τo dictionary με τα embeddings
embeddings_index = dict()
f = open("glove.6B.100d.txt", encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefficients
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))



# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)


#δημιουργία του νευρωνικού model
model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#κάνομυε compile το model μας
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# summarize the model
print(model.summary())


#κάνουμε split το dataset σε train και test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_emails, labels, test_size=0.25, random_state=42)


#κάνουμε fit το model μας
model.fit(X_train, y_train, epochs=10, verbose=0)

# evaluation του model μας
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict_classes(X_test)
cf_matrix=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix\n\n {} \n\n {}".format(cf_matrix,classification_report(y_test,y_pred)))




