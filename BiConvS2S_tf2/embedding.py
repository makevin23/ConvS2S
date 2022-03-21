import numpy as np
import tensorflow as tf
import tensorflow.python.ops.numpy_ops.np_config as np_config



class Embed():
    def __init__(self, vocab: dict, embedding_size: int, batch_size: int) -> None:
        self.vocab = vocab
        self.embedding_size = embedding_size
        self.vocab_size = len(vocab)
        self.batch_size = batch_size

    def embed_position(self, X):
        position_vectors = []
        for sentence in X:
            current_vector = []
            for i in range(sentence.size):
                current_vector.append(i)
            position_vectors.append(np.asarray(current_vector))
        self.position_vectors = np.asarray(position_vectors)

    def generate_input_pairs(self, training_sentences):
       
        words = []
        previous_and_following_words = []

        for sentence in training_sentences:

            # For each sentence in the training set
            for i in range(sentence.size):

                # For each word in the training set
                # Append the word before it and the 
                # word after it to the label (If either exist)
                if(i + 1 != len(sentence)):
                    words.append(sentence[i]) 
                    previous_and_following_words.append(sentence[i + 1])
                if(i - 1 != -1):
                    words.append(sentence[i])
                    previous_and_following_words.append(sentence[i - 1])

        # Convert them into numpy arrays
        self.X = np.asarray(words)
        self.Y = np.asarray(previous_and_following_words)
        self.Y = np.expand_dims(self.Y, 1)
    


    def train_embedder(self, X):

        # Generate the input-output pairs
        self.generate_input_pairs(X)
        self.embedding_words = tf.random.uniform([self.vocab_size, self.embedding_size], -1, 1)
        nce_weights = np.ndarray(tf.random.uniform([self.vocab_size, self.embedding_size], stddev = 1/np.sqrt(self.embedding_size)), name = "Embedding_Layer")
        nce_biases = np.ndarray(tf.zeros([self.vocab_size]), name = "Embedding_Biases")
        n_batches = len(self.X) // self.batch_size
        train_inputs = np.ndarray(shape = [self.batch_size], dtype = int, name = "Dictionary_Input")
        train_labels = np.ndarray(shape = [self.batch_size, 1], dtype = int, name = "Embedded_Output")



    def get_embedding(self, X):
        embedded_words = []
        
        return 1

    def generate_embeddings(self, training_sentences):
        return (self.get_embedding(training_sentences) + self.get_position_embedding(training_sentences))




    def one_hot_encoding(self, X):
        one_hot = np.zeros((X.shape[0], X.shape[1], self.vocab_size))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                one_hot[i, j, X[i, j]] = 1
        return one_hot
