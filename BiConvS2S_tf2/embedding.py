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
    def train_step(self, nce_weights, nce_biases, train_inputs, train_labels):
        embed = tf.nn.embedding_lookup(self.embedding_words, train_inputs)

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
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1)

        optimizer.minimize(loss, var_list=self.embedding_words)


    def train_embedder(self, X):
        self.generate_input_pairs(X)

        self.embedding_words = tf.Variable(tf.random.uniform(
            [self.vocab_size, self.embedding_size], -1, 1), name="Embedder")

        nce_weights = tf.Variable(tf.random.truncated_normal(
            [self.vocab_size, self.embedding_size], stddev=1/np.sqrt(self.embedding_size)), name="Embedding_Layer")
        nce_biases = tf.Variable(
            tf.zeros([self.vocab_size]), name="Embedding_Biases")
        n_batches = len(self.X) // self.batch_size

        # training part
        for _i in range(100):
            index_ = np.arange(0, self.X.shape[0])
            np.random.shuffle(index_)
            index = []
            for i in range(n_batches):
                    index.append(index_[(self.batch_size*i):(self.batch_size*(i + 1))])
            for batch in index:
                train_inputs = self.X[batch]
                train_labels = self.Y[batch]
                self.train_step(nce_weights, nce_biases, train_inputs, train_labels)
                

    @tf.function
    def look_up_word(self, word_tf):
        return tf.nn.embedding_lookup(self.embedding_words, word_tf, name = "Look_Up_Function")
                

    def get_embedding(self, X):
        embedded_words = []
        for sentence in X:
                vecc = []
                sentence = [int(x) for x in sentence]
                for word in sentence:
                    vecc.append(self.look_up_word(word))
                embedded_words.append(np.asarray(vecc))
        return(np.asarray(embedded_words))

    @tf.function
    def look_up_position(self, position_tf):
        return tf.nn.embedding_lookup(self.embedding_words, position_tf, name = "Look_Up_Function_pos")

    def get_position_embedding(self, X):
        position = []
        self.embed_position(X)
        for vec in self.position_vectors:
                temp = []
                for value in vec:
                    temp.append(self.look_up_position(value))
                position.append(np.asarray(temp))
        return np.asarray(position)


    def generate_embeddings(self, X):
        return (self.get_embedding(X) + self.get_position_embedding(X))


    @tf.function
    def index_vocab(self, input_vec):
        vocab = tf.matmul(
            tf.nn.l2_normalize(self.embedding_words, axis = 1),
            tf.nn.l2_normalize(input_vec, axis = 1),
            transpose_b = True
        )
        index_ = tf.argmax(vocab, axis = 1)
        pass

    # TODO: write unit test for generate_vocab
    def generate_vocab(self, embedding_vec):
        return self.index_vocab(embedding_vec)

    def one_hot_encoding(self, X):
        one_hot = np.zeros((X.shape[0], X.shape[1], self.vocab_size))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                one_hot[i, j, X[i, j]] = 1
        return one_hot