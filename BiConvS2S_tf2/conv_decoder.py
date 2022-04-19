import tensorflow as tf
import numpy as np

# How does the decoder work?:
# 1. The decoder takes the decoder inputs (Which are just the targets with the 
# last element removed) and the target values (which is just the target vector 
# with the first element removed) and the encoder outputs and the encoder 
# attention
# 2. It then converts the input from the embedding space to the size of the kernel 
# (the hidden space).
# 3. The convolutional layer runs its course and we get a matrix which has 
# dimensionality twice the hidden space. We then run the GLU activation unit
# on the output. We get back a matrix in the hidden space.
# 4. This GLU output is converted back into the embedded space. The GLU output 
# is then dot multiplied with the encoder attention. (Why? That is 
# how we define the attention in the original paper) We take the softmax of 
# the resulting matrix. 
# 5. The attention output is the matrix multiplication of the encoder outputs 
# and the vector we got from the above step. We then scale this attention and 
# convert it back to hidden space.
# 6. The process steps 3-5 are repeated as per the number of layers in the 
# convolutional network. The final output of the convolutional network is 
# converted back into the embedding space.
# 7. This output is converted into vocab space, and the softmax is taken. 
# 8. This output is the final output of the network. We predict the correct 
# words based on this. 
# 9. In case of calculating the output in the forward direction, the decoder 
# inputs are a matrix of dimensions (no_of_input_sentences * (max_decoder_length + 2)). 
# We initialize the first column of this matrix with a all ones vector. We feed 
# this matrix to the decoder and we get thee next column in the decoder 
# input matrix. We keep feeding this till we get a <end> token. This is the output 
# prediction.

# The code below will simply be divided in terms of the steps given above
class ConvDecoder(tf.keras.Model):
    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, num_layers, dropout, is_training = True):
        super(ConvDecoder, self).__init__()
        # Initialize the class variables
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.p = dropout
        self.dropout_layer = tf.keras.layers.Dropout(1-self.p)
        self.is_training = is_training
        self.kernel_size = [5, self.hidden_size]

        # Define the various placeholders and layers
        self.dense_layer_1 = tf.keras.layers.Dense(self.hidden_size, activation="relu", name = "decoder_dense_layer_1")
        self.dense_layer_2 = tf.keras.layers.Dense(self.embedding_size, activation="relu", name = "decoder_dense_layer_2")
        self.dense_layer_3 = tf.keras.layers.Dense(self.vocab_size, activation="relu", name = "decoder_dense_layer_3")
        self.layer_conv_embedding = tf.keras.layers.Dense(self.embedding_size, activation=tf.keras.activations.softmax, name = "decoder_conv_embedding")
        self.layer_embedding_conv = tf.keras.layers.Dense(self.hidden_size, activation=tf.keras.activations.softmax, name = "decoder_embedding_conv")
        self.conv_layer = tf.keras.layers.Conv2D(filters=2 * self.hidden_size, kernel_size=self.kernel_size, padding="same", name="decoder_conv_layer")

    def for_decoder(self, encoder_outputs, encoder_attention):
        # self.input_x = tf.compat.v1.placeholder(dtype = tf.float32, shape = [None, self.max_length - 1, self.embedding_size], name = "Decoding_Input")
        # Step 1:
        self.input_x = self.dropout_layer(self.input_x, training=True)
        temp = tf.reshape(self.input_x, [tf.shape(self.input_x)[0]*self.input_x.shape[1], self.input_x.shape[2]])

        # Step 2:
        dl1_out_ = self.dense_layer_1(temp)
        dl1_out_ = tf.reshape(dl1_out_,  [tf.shape(dl1_out_)[0]/(self.max_length - 1), self.max_length - 1, self.hidden_size])
        layer_output = dl1_out_
        for _ in range(self.num_layers):

            # Step 3
            residual_output = layer_output
            dl1_out = self.dropout_layer(layer_output, training=True)
            dl1_out = tf.expand_dims(layer_output, axis = 0)
            conv_layer_output = self.conv_layer(dl1_out)
            glu_output = conv_layer_output[:, :, :, 0:self.hidden_size] * tf.nn.sigmoid(conv_layer_output[:, :, :, self.hidden_size:(2*self.hidden_size)])
            glu = tf.squeeze(glu_output, axis = 0)
            layer_output = (glu + residual_output) * np.sqrt(0.5)
            shape_out = tf.shape(layer_output)

            # Step 4:
            layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0]*layer_output.shape[1], layer_output.shape[2]])
            post_glu_output = self.layer_conv_embedding(layer_output)
            post_glu_output = tf.reshape(post_glu_output, [tf.shape(post_glu_output)[0]/(self.max_length - 1), self.max_length - 1, self.embedding_size])
            encoder_attention_logits = tf.matmul( post_glu_output, tf.transpose(encoder_attention, perm = [0, 2, 1]), name = "Encoder_Attention_MatMul")
            encoder_attention_output = tf.nn.softmax(encoder_attention_logits, axis = 2)

            # Step 5:
            attention_output = tf.matmul(encoder_attention_output, encoder_outputs, name = "Attention_Output")
            attention_output = attention_output * (encoder_outputs.shape.as_list()[2] * np.sqrt(2 / encoder_outputs.shape.as_list()[2]))
            layer_output = tf.reshape(attention_output, [tf.shape(attention_output)[0]*attention_output.shape[1], attention_output.shape[2]])
            layer_output = self.layer_embedding_conv(layer_output)
            layer_output = tf.reshape(layer_output, shape_out)
        
        # Step 6:
        layer_output = (layer_output + dl1_out_) * np.sqrt(0.5)
        layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0]*layer_output.shape[1], layer_output.shape[2]])
        output = self.dense_layer_2(layer_output)

        # Step 7:
        self.prob_output = self.dense_layer_3(output)
        self.prob_output = tf.reshape(self.prob_output, [tf.shape(self.prob_output)[0]/(self.max_length - 1), self.max_length - 1, self.vocab_size])
        self.prob_output = tf.nn.softmax(self.prob_output, 2)
        return (self.prob_output)
    
    def rev_decoder(self):
        self.input_x_rev = self.dropout_layer(self.input_x_rev, training=True)
        temp = tf.reshape(self.input_x_rev, [tf.shape(self.input_x_rev)[0]*self.input_x_rev.shape[1], self.input_x_rev.shape[2]])
        dl1_out_ = self.dense_layer_1(temp)
        dl1_out_ = tf.reshape(dl1_out_, [tf.shape(dl1_out_)[0]/self.max_length, self.max_length, self.hidden_size])
        layer_output = dl1_out_
        for _ in range(self.num_layers):

            # Step 3:
            residual_output = layer_output
            self.checker = layer_output
            dl1_out = self.dropout_layer(layer_output, training=True)
            dl1_out = tf.expand_dims(dl1_out, axis = 0)
            conv_layer_output = self.conv_layer(dl1_out)
            glu_output = conv_layer_output[:, :, :, 0:self.hidden_size] * tf.nn.sigmoid(conv_layer_output[:, :, :, self.hidden_size:(2*self.hidden_size)])
            glu = tf.squeeze(glu_output, axis = 0)
            layer_output = (glu + residual_output) * np.sqrt(0.5)
        
        # Step 5:
        layer_output = tf.reshape(layer_output, [tf.shape(layer_output)[0]*layer_output.shape[1], layer_output.shape[2]])
        self.decoder_output_ = self.dense_layer_2(layer_output)
        self.decoder_output_ = tf.reshape(self.decoder_output_, [tf.shape(self.decoder_output_)[0]/self.max_length, self.max_length, self.decoder_output_.shape[1]])

        # Step 6:
        self.decoder_attention_ = (self.decoder_output_ + self.input_x_rev) * np.sqrt(0.5)
        
        return self.decoder_output_, self.decoder_attention_
