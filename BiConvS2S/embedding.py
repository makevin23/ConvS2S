import numpy as np
import tensorflow as tf
import tensorflow.python.ops.numpy_ops.np_config as np_config

# The Embed class is used to generate the word embeddings. 
# The class using tri-grams for context. Each tri-gram can be 
# defined as (a-1, a, a + 1). Here 'a' is the word under 
# consideration. Hence, two training pairs are generated:
# (a, a - 1) and (a, a + 1)
# This is then used to generate the embeddings for each pair.

# The Loss use is NCE (Noise Contrastive Estimation) Loss
# A brief explaination of the NCE Loss: 
# 1. Assume that each word makes up its own class. We calculate
# the probability Pr ( Label | Class )
# 2. We then contrast it with Pr ( Label | Not Class ) where 
# Not Class is any other class except the class under consideration.
# This probability is then maximised in the NCE Loss.

class Embed():

    # Init function
    def __init__(self, vocab, embedding_size, batch_size):
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.vocab_size = len(vocab)
        self.batch_size = batch_size

    # The position embedding is a concept unique to ConvS2S. It is 
    # simply the embeddings generated by a vector with the absolute
    # location of the elements. (a[i] = i)
    def embed_position(self, X):
        position_vectors = []
        for sentence in X:
            current_vector = []
            for i in range(sentence.size):
                current_vector.append(i)
            position_vectors.append(np.asarray(current_vector))
        self.position_vectors = np.asarray(position_vectors)

    # This function generates pairs for training
    def generate_input_pairs(self, X):
        np_config.enable_numpy_behavior()
        X_ = []
        Y_ = []
        for sentence in X:
            # For each sentence in the training set
            for i in range(sentence.size):

                # For each word in the training set
                # Append the word before it and the 
                # word after it to the label (If either exist)
                if(i + 1 != len(sentence)):
                    X_.append(sentence[i]) 
                    Y_.append(sentence[i + 1])
                if(i - 1 != -1):
                    X_.append(sentence[i])
                    Y_.append(sentence[i - 1])

        # Convert them into numpy arrays
        self.X = np.asarray(X_)
        self.Y = np.asarray(Y_)
        self.Y = np.expand_dims(self.Y, 1)
    
    # Embedding training module
    def train_embedder(self, X):

        # Generate the input-output pairs
        self.generate_input_pairs(X)

        # Initialize all the variables and placeholders
        self.embedding_words = tf.random.uniform([self.vocab_size, self.embedding_size], -1, 1)
        nce_weights = tf.Variable(tf.compat.v1.random.truncated_normal([self.vocab_size, self.embedding_size], stddev = 1/np.sqrt(self.embedding_size)), name = "Embedding_Layer")
        nce_biases = tf.Variable(tf.zeros([self.vocab_size]), name = "Embedding_Biases")
        n_batches = len(self.X) // self.batch_size
        train_inputs = tf.compat.v1.placeholder(tf.int32, shape = [self.batch_size], name = "Dictionary_Input")
        train_labels = tf.compat.v1.placeholder(tf.int32, shape = [self.batch_size, 1], name = "Embedded_Output")

        # The function embedding_lookup gives us the embedding based on the input pair
        embed = tf.nn.embedding_lookup(self.embedding_words, train_inputs)

        # Calculate the NCE Loss using the in-built model tf.nn.nce_loss
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights = nce_weights,
                biases = nce_biases,
                labels = train_labels,
                inputs = embed,
                num_sampled = 8,
                num_classes = self.vocab_size
            )
        )

        # Optimize the loss
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
        with tf.compat.v1.Session() as sess:

            # Start the session and use mini-batch stochastic 
            # gradient descent to train the model
            sess.run(tf.compat.v1.global_variables_initializer())
            for _i in range(100):
                index_ = np.arange(0, self.X.shape[0])
                np.random.shuffle(index_)
                index = []
                for i in range(n_batches):
                    index.append(index_[(self.batch_size*i):(self.batch_size*(i + 1))])
                for batch in index:
                    feed_dict = {train_inputs : self.X[batch], train_labels : self.Y[batch]}
                    _, cur_loss = sess.run([optimizer, loss], feed_dict = feed_dict)


    @tf.function
    def lookup_in_embedding(self, word_ids):
        return tf.nn.embedding_lookup(self.embedding_words, word_ids, name="Look_Up_Function")

    # Use the trained embedding function to 
    # generate the embeddings of the input sentences
    def get_embedding(self, X):
        embedded_words = []
        # word_tf = tf.compat.v1.placeholder(dtype = tf.int32, shape = None)
        # look_up = tf.nn.embedding_lookup(self.embedding_words, word_tf, name = "Look_Up_Function")
        # with tf.compat.v1.Session() as sess:
        #     sess.run(tf.compat.v1.global_variables_initializer())
        #     for sentence in X:
        #         vecc = []
        #         for word in sentence:
        #             vecc.append(sess.run(look_up, feed_dict = { word_tf : word}))
        #         embedded_words.append(np.asarray(vecc))

        for sentence in X:
            look_up = self.lookup_in_embedding(sentence)
            # look_up = tf.nn.embedding_lookup(self.embedding_words, sentence).numpy()
            vecc = [look_up]
            embedded_words.append(np.asarray(vecc))

        return(np.asarray(embedded_words))

    
    
    # Generate the position embeddings of the input sentences
    def get_position_embedding(self, X):
        position = []
        self.embed_position(X)
        position_tf = tf.compat.v1.placeholder(dtype = tf.int32, shape = None)
        look_up = tf.nn.embedding_lookup(self.embedding_words, position_tf, name = "Look_Up_Function_pos")
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for vec in self.position_vectors:
                temp = []
                for value in vec:
                    temp.append(sess.run(look_up, feed_dict = {position_tf : value}))
                position.append(np.asarray(temp))
        return np.asarray(position)

    # Add the embeddings of the words and the 
    # position embeddings to get the final embeddings
    def generate_embeddings(self, X):
        return (self.get_embedding(X) + self.get_position_embedding(X))
    
    # Given a embedding vector return the word
    def generate_vocab(self, embedding_vec):
        input_vec = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, self.embedding_size], name = "Inverse_Input")

        # The word is just the argmax of the multiplication 
        # of the embedding matrix and the input embedding pair
        vocab = tf.matmul(
            tf.nn.l2_normalize(self.embedding_words, axis = 1),
            tf.nn.l2_normalize(input_vec, axis = 1),
            transpose_b = True
        )
        index_ = tf.argmax(vocab, axis = 1)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            index = sess.run(index_, feed_dict = { input_vec : embedding_vec })
        return index

    # Using the vocabulary, return a one-hot encoded vector 
    # for a given word vector or a set of sentencec
    def one_hot_encoding(self, X):
        one_hot = np.zeros((X.shape[0], X.shape[1], self.vocab_size))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                one_hot[i, j, X[i, j]] = 1
        return one_hot

    def export_embedder(self, file_path):
        with open(file_path+"/embedding_words.npy", "wb") as f:
            np.save(f, self.embedding_words)

    def load_embedding_words(self, embedding_words_file):
        with open(embedding_words_file+"/embedding_words.npy", "rb") as f:
            self.embedding_words = np.load(f)


