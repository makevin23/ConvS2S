
# Convolutional Sequence-to-Sequence Learning
# Author : Tanmayan Pande
# Paper : https://arxiv.org/pdf/1705.03122.pdf
#####################################################################

import os
import pickle
import tensorflow as tf
import numpy as np
import string
import embedding
import conv_encoder
import conv_decoder
import training

######################################################################
# Define the input and the target pairs.
# examples : Input
# target : Corresponding Outputs
######################################################################
examples = [
    "Hi, how are you?",
    "Can you turn on the fan?",
    "Can you tell me the weather?"
]

target = [
    "I am fine, thank you",
    "Turning on the fan on a medium speed",
    "The weather is 19 and sunny"
]

# TODO: load training data from data.en and data.sparql

trained_model_dir = "BiConvS2S_tf2"
ckpt_dir = os.path.join(trained_model_dir, "ckpt")
pkl_dir = trained_model_dir + "/pkl"

#######################################################################
# Define the properties of the example corpus and generate the 
# vocabulary.
# train_x = Input training num vectors built using the generated 
#           vocabulary
# train_y = Input training label num vectors built using the 
#           generated vocabulary
#######################################################################

max_input_length = 0
max_target_length = 0
word_to_index = {'<unk>' : 0, '<end>' : 1}
index_to_word = {0 : '<unk>' , 1 : '<end>'}
train_x = []
train_y = []

# Procedure to generate the vocabulary:
# For all sentences in the corpus:
for sentence in examples:

    # Remove the punctuation marks
    removed_punc = sentence.translate(sentence.maketrans("", "", string.punctuation))

    # Convert all the letters to lowercase
    removed_punc = removed_punc.lower()

    # Split the string and generate the list
    split_sent = list(removed_punc.split())

    # Update max_length if necessary
    if (len(split_sent)) > max_input_length:
        max_input_length = len(split_sent)

    vectorized = []

    # Check for every word in split_sent
    for word in split_sent:

        # If not in vocabulary, add the word to the vocabulary

        if not word_to_index.get(word):
            word_to_index[word] = len(word_to_index)
            index_to_word[len(index_to_word)] = word

            # Update the vectorized input
            vectorized.append(word_to_index[word])
        else:

            # Else, just append the vectorized word to 
            # the vector which represents the sentence
            vectorized.append(word_to_index[word])
    vectorized.append(1)
    train_x.append(vectorized)

# Follow a similar process for the target corpus

for sentence in target:
    removed_punc = sentence.translate(sentence.maketrans("", "", string.punctuation))
    removed_punc = removed_punc.lower()
    split_sent = list(removed_punc.split())
    if (len(split_sent)) > max_input_length:
        max_target_length = len(split_sent)

    vectorized = []
    for word in split_sent:
        if not word_to_index.get(word):
            word_to_index[word] = len(word_to_index)
            index_to_word[len(index_to_word)] = word
            vectorized.append(word_to_index[word])
        else:
            vectorized.append(word_to_index[word])
    vectorized.append(1)
    vectorized.insert(0, 1)
    train_y.append(vectorized)

# Pad the examples and the labels with zero to ensure equal length
train_x = np.asarray([np.pad(example, [0, max_input_length - len(example) + 2], mode = 'constant') for example in train_x]).astype(int)
train_y = np.asarray([np.pad(example, [0, max_target_length - len(example) + 2], mode = 'constant') for example in train_y]).astype(int)

os.mkdir(pkl_dir)
with open(pkl_dir+'/word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f)

with open(pkl_dir+'/index_to_word.pkl', 'wb') as f:
    pickle.dump(index_to_word, f)

with open(pkl_dir+'/max_input_length.pkl', 'wb') as f:
    pickle.dump(max_input_length, f)

with open(pkl_dir+'/max_target_length.pkl', 'wb') as f:
    pickle.dump(max_target_length, f)

# Train the Embedder Network on the examples
embedder = embedding.Embed(word_to_index, 512, 16)
embedder.train_embedder(train_x)
np.save(pkl_dir+'/embedding_words.npy', embedder.embedding_words)
# train_embeddings = embedder.generate_embeddings(train_x)
# label_embeddings = embedder.generate_embeddings(train_y)

# Build the encoder and the decoder networks
Encoder = conv_encoder.ConvEncoder(len(index_to_word), max_input_length + 2, 128, 512, 1, 1)
Decoder = conv_decoder.ConvDecoder(len(index_to_word), max_target_length + 2, 128, 512, 1, 1)

# Pass them to the trainer and train the CNN
trainer = training.Translator(Encoder, Decoder, embedder, word_to_index, index_to_word, ckpt_dir)
trainer(inputs = train_x, targets = train_y, is_training = True)

