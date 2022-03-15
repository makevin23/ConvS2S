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

    @tf.function
    def optimize_loss(self,X):
        self.generate_input_pairs(X)

        self.embedding_words = tf.Variable(tf.random.uniform(
            [self.vocab_size, self.embedding_size], -1, 1), name="Embedder")

        nce_weights = tf.Variable(tf.random.truncated_normal(
            [self.vocab_size, self.embedding_size], stddev=1/np.sqrt(self.embedding_size)), name="Embedding_Layer")
        nce_biases = tf.Variable(
            tf.zeros([self.vocab_size]), name="Embedding_Biases")
        train_inputs = tf.keras.Input(
            dtype=tf.int32, shape=[self.batch_size], name="Dictionary_Input")
        train_labels = tf.keras.Input(
            dtype=tf.int32, shape=[self.batch_size], name="Embedded_Output")

        embed = tf.nn.embedding_lookup(self.embedding_words, train_inputs)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=embed,
                num_sampled=8,
                num_classes=self.vocab_size
            )
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1).minimize(loss, var_list = [train_inputs, train_labels])



    def train_embedder(self, X):
        n_batches = len(self.X) // self.batch_size
        train_inputs = np.ndarray(shape = [self.batch_size], dtype = int, name = "Dictionary_Input")
        train_labels = np.ndarray(shape = [self.batch_size, 1], dtype = int, name = "Embedded_Output")


        # training part
        # 100 epochs
        for _i in range(100):
            index_ = np.arange(0, self.X.shape[0])
            np.random.shuffle(index_)
            index = []
            for i in range(n_batches):
                    index.append(index_[(self.batch_size*i):(self.batch_size*(i + 1))])
            for batch in index:
                self.optimize_loss(X)
                

    @tf.function
    def look_up_word(self, word_tf):
        return tf.nn.embedding_lookup(self.embedding_words, word_tf, name = "Look_Up_Function")
                

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
