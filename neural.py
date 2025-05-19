
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Embedding
import json

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# Load the dataset
dataset = pd.read_csv('spam_or_not_spam.csv')



# Convert email column to string type
dataset.email = dataset.email.astype(str)

# Extract emails and labels
emails = dataset["email"].values
labels = dataset["label"].values
print(len(emails))

# Create tokenizer and fit on emails
tokenizer = Tokenizer()
tokenizer.fit_on_texts(emails)
vocab_size = len(tokenizer.word_index) + 1

# Integer encode the emails
encoded_emails = tokenizer.texts_to_sequences(emails)
print(vocab_size)

# Pad sequences to ensure uniform lengthh
max_length = 20
padded_emails = pad_sequences(encoded_emails, maxlen=max_length, padding='post')
print(len(padded_emails))

# Load GloVe word embeddings
embeddings_index = dict()
f = open("glove.6B.100d.txt", encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefficients = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefficients
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# Create embedding matrix for words in our vocabulary
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

# Build the neural network model
model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=20, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Print model summary
print(model.summary())

# Split dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(padded_emails, labels, test_size=0.25, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=0)

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))

# Generate classification metrics
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict_classes(X_test)
cf_matrix=confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix\n\n {} \n\n {}".format(cf_matrix,classification_report(y_test,y_pred)))



tokenizer_json = tokenizer.to_json()
with open("tokenizer.json", "w") as f:
    f.write(tokenizer_json)

pipeline_metadata = {
    "max_length": max_length,  
    "metrics": {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    },
    "config": {
        "vocab_size": vocab_size,
        "classes": ["not_spam", "spam"]  
    }
}

with open("pipeline_metadata.json", "w") as f:
    json.dump(pipeline_metadata, f)

model.save('spam_classifier.keras')
