import tensorflow as tf
import numpy as np
import os


# The Translator class trains the network translating between languages

class Translator():
    def __init__(self, encoder, decoder, embedder, vocab, inverse_vocab, ckpt_dir, learning_rate = 0.001, batch_size = 16, epochs = 100):
        self.encoder = encoder
        self.decoder = decoder
        self.embedder = embedder
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        self.loss = tf.nn.softmax_cross_entropy_with_logits
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr)
        self.ckpt_dir = ckpt_dir
        
# The __call__() enables use to build a callable object.
    def __call__(self, inputs, targets = None, is_training = False):

        # Check if the model is trained or not
        if is_training:

            # If not, check if targets are passed or not
            if targets.all() is None:
                raise ValueError("The target cannot be empty if you are training")
            else:
                # Start training
                self.start_train(inputs, targets)
        else:

            # If model is trained, just return the outputs of the network
            output = self.start_eval(inputs)
            return output


    def init_train_step(self, input_x, decoder_input, target, encoder_input_rev, input_x_rev, target_rev):
        self.encoder.X = input_x
        self.decoder.input_x = decoder_input
        self.target_pl = target
        self.encoder.X_rev = encoder_input_rev
        self.decoder.input_x_rev = input_x_rev
        self.target_pl_rev = target_rev

    def train_step_for(self):
        with tf.GradientTape() as tape:
            encoder_output, encoder_attention = self.encoder.for_encoder()
            prob_output = self.decoder.for_decoder(encoder_output, encoder_attention)
            loss_fxn_for = tf.reduce_mean(self.loss(labels = self.target_pl, logits = prob_output))
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss_fxn_for, variables)
            self.optimizer.apply_gradients((gradients, variables) for (gradients, variables) in zip(gradients, variables) if gradients is not None)
        return loss_fxn_for

    def train_step_rev(self):
        with tf.GradientTape() as tape:
            decoder_output, decoder_attention = self.decoder.rev_decoder()
            prob_output_rev = self.encoder.rev_encoder(decoder_output, decoder_attention)
            loss_fxn_rev = tf.reduce_mean(self.loss(labels = self.target_pl_rev, logits = prob_output_rev))
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss_fxn_rev, variables)
            # self.optimizer.apply_gradients(zip(gradients, variables))
            self.optimizer.apply_gradients((gradients, variables) for (gradients, variables) in zip(gradients, variables) if gradients is not None)
        return loss_fxn_rev

# Training the model:
    def start_train(self, inputs, targets):

        # Generate the batches over the training data
        index_ = np.arange(inputs.shape[0])
        np.random.shuffle(index_)
        index = []
        n_batches = np.floor(index_.size / self.batch_size).astype(int)
        for i in range(n_batches):
            index.append(index_[(i*self.batch_size):((i + 1)*self.batch_size)])
        # If the size of the training data is not entirely divisible by the 
        # batch size, make the last batch with the last (batch_size) inputs
        if(index_.size % self.batch_size != 0):
            index.append(index_[-self.batch_size::])
        # Generate the embeddings for the training data and the labels
        train_embeddings = self.embedder.generate_embeddings(inputs)
        label_embeddings = self.embedder.generate_embeddings(targets)


        # Define the network, the optimizer and the loss function
        loss = []
        
        # Start training for the number of epochs given
        for i in range(self.epochs):
            loss_epoch = []
            for batch in index:

                # Generate the inputs and the targets from the embeddings generated
                input_x = train_embeddings[batch, :, :]
                decoder_input = label_embeddings[batch, :-1, :]
                input_x_rev = label_embeddings[batch, :, :]
                encoder_input_rev = train_embeddings[batch, :-1, :]

                # One Hot encode the targets for loss calculations
                target = self.embedder.one_hot_encoding(targets[batch, 1:])
                target_rev = self.embedder.one_hot_encoding(inputs[batch, 1:])

                # Run the minimization and the loss function op
                self.init_train_step(input_x, decoder_input, target, encoder_input_rev, input_x_rev, target_rev)
                loss_val_for = self.train_step_for()
                loss_val_rev = self.train_step_rev()
                loss_epoch.append([loss_val_for, loss_val_rev])
            loss.append(np.mean(loss_epoch, axis = 0))

            # Print the average loss after each epoch
            print("EPOCH " + str(i + 1) + " COMPLETED !")
            print("LOSS : " + str(loss[i]))
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, encoder=self.encoder,decoder=self.decoder)
        # save the model
        checkpoint.save(file_prefix=os.path.join(self.ckpt_dir, "ckpt"))
        print("Model saved !")
        # Create a folder to store the weights of the model
        print("Finished Optimization")


    @tf.function
    def output_step(self, encoder_X, decoder_input_x):
        self.encoder.X = encoder_X
        self.decoder.input_x = decoder_input_x
        encoder_output, encoder_attention = self.encoder.for_encoder()
        return self.decoder.for_decoder(encoder_output, encoder_attention)


# Run the network for the given inputs
    def start_eval(self, inputs):

        decoder_inputs = np.zeros((inputs.shape[0], self.decoder.max_length - 1))
        decoder_inputs[:, 0] = np.ones((inputs.shape[0]))
        next_decoder_output = None
        i = 1

            # Start the session. While the output_length is less than the max_length for the decoder, run the netwrok
        while(next_decoder_output is None or next_decoder_output[0, 0] is not 1) and len(decoder_inputs[0, :]) < self.decoder.max_length and i < self.decoder.max_length - 1:

            output = self.output_step(self.embedder.generate_embeddings(inputs), self.embedder.generate_embeddings(decoder_inputs))

            # Convert the output into vocab indices
            next_decoder_output = np.argmax(output, axis = 2)

            # Update the decoder inputs to reflect this prediction.
            decoder_inputs[:, i] = next_decoder_output[:, i]
            i += 1

        output_strings = []

        # Convert the index vectors into words and subsequently, sentences
        for i in range(decoder_inputs.shape[0]):
            string_temp = []
            for j in range(1, decoder_inputs.shape[1]):
                string_temp.append(self.inverse_vocab[decoder_inputs[i, j]])
            add_string = ' '.join(string_temp)
            output_strings.append(add_string)
        return output_strings
