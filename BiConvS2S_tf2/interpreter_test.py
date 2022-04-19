import os
import pickle
import tensorflow as tf
import numpy as np
import string
import embedding
import conv_encoder
import conv_decoder
import training

examples = [
    "Did <A> study at the <B> university?"
]

target = [
    "I am fine, thank you",
    "Turning on the fan on a medium speed",
    "The weather is 19 and sunny"
]

trained_model_dir = "BiConvS2S_tf2"
pkl_dir = trained_model_dir + "/pkl"
ckpt_dir = trained_model_dir + "/ckpt"

# load parameters
with open(pkl_dir+'/max_input_length.pkl', 'rb') as f:
    max_input_length = pickle.load(f)

with open(pkl_dir+'/max_target_length.pkl', 'rb') as f:
    max_target_length = pickle.load(f)

# load word index
with open(pkl_dir+'/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)

with open(pkl_dir+'/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)

train_x = []

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
        try:
            # the vector which represents the sentence
            vectorized.append(word_to_index[word])
        except:
            # if the word is not in the vocabulary, then use the <unk> token
            vectorized.append(word_to_index['<unk>'])
    vectorized.append(1)
    train_x.append(vectorized)

# Follow a similar process for the target corpus

# Pad the examples and the labels with zero to ensure equal length
train_x = np.asarray([np.pad(example, [0, max_input_length - len(example) + 2], mode = 'constant') for example in train_x]).astype(int)

embedder = embedding.Embed(word_to_index, 512, 16)
embedder.embedding_words = np.load(pkl_dir+'/embedding_words.npy')

# Build the encoder and the decoder networks
Encoder = conv_encoder.ConvEncoder(len(index_to_word), max_input_length + 2, 128, 512, 1, 1)
Decoder = conv_decoder.ConvDecoder(len(index_to_word), max_target_length + 2, 128, 512, 1, 1)
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    encoder=Encoder,
                                    decoder=Decoder)

checkpoint.restore(tf.train.latest_checkpoint(trained_model_dir))

trainer = training.Translator(Encoder, Decoder, embedder, word_to_index, index_to_word, ckpt_dir)

print(trainer(inputs = train_x))