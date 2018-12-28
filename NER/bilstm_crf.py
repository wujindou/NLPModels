#coding:utf-8

#coding:utf-8
import tensorflow as tf
import sys


from numpy.random import seed
import numpy as np
np.random.seed(7)
import tensorflow as tf
tf.set_random_seed(7)
class Model(object):

	def __init__(self, input_steps, embedding_size, hidden_size, vocab_size, slot_size,batch_size=16, n_layers=1,embedding_matrix=None):
		self.input_steps = input_steps
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.learning_rate = 0.001
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.slot_size = slot_size
		self.embedding_matrix = embedding_matrix
		self.encoder_inputs = tf.placeholder(tf.int32, [batch_size, input_steps],name='encoder_inputs')

		self.encoder_inputs_actual_length = tf.placeholder(tf.int32, [batch_size],name='encoder_inputs_actual_length')
		self.decoder_targets = tf.placeholder(tf.int32, [batch_size, input_steps], name='decoder_targets')
		self.dropout = tf.placeholder_with_default(1.0, shape=())
		self.global_step = tf.Variable(0, trainable=False, name="global_step")



	def build_graph(self):

		if self.embedding_matrix is None:

			#embedding = tf.get_variable("embedding",[self.vocab_size,self.embedding_size],initializer=tf.random_normal_initializer(stddev=0.1),dtype=tf.float32)
			embedding = tf.get_variable("embedding",[self.vocab_size,self.embedding_size],dtype=tf.float32)#embedding = tf.get_variable("embedding",[self.vocab_size,self.embedding_size],initializer=tf.random_normal_initializer(stddev=0.1),dtype=tf.float32)
			#embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W")
		else:
			embedding = tf.get_variable(name='W',shape=self.embedding_matrix.shape,initializer=tf.constant_initializer(self.embedding_matrix))

		self.inputs = tf.nn.embedding_lookup(embedding,self.encoder_inputs)
		self.inputs = tf.nn.dropout(self.inputs, self.dropout)

		self.bilstm_encoder()

		# self.intent_loss()
		self.crf_layer()

	def bilstm_encoder(self):
		with tf.variable_scope("layer1"):
			#cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
			cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size)
			#cell_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
			#cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,input_keep_prob=self.dropout,output_keep_prob=self.dropout)
			cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size)
			#cell_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
			#cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,input_keep_prob=self.dropout,output_keep_prob=self.dropout)
			(output_fw, output_bw), final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.inputs,self.encoder_inputs_actual_length,  dtype=tf.float32)
			context_rep = tf.concat([output_fw, output_bw], axis=-1)
			context_rep = tf.nn.dropout(context_rep,self.dropout)
			# self.sent_b_attn(context_rep)
			first_dense_output= tf.layers.dense(context_rep,100,activation=tf.nn.tanh)# self.sent_b_attn(context_rep)
		self.logits = tf.layers.dense(first_dense_output, self.slot_size)


	def sent_b_attn(self,input):
		with tf.variable_scope("attn_b"):
			U_it = tf.contrib.layers.fully_connected(input, self.hidden_size * 2,activation_fn=tf.nn.tanh)
			# print(U_it)

			U_context = tf.get_variable(name="u_context",shape=[self.hidden_size*2],initializer=tf.contrib.layers.xavier_initializer())


			alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(U_it,U_context),axis=2,keep_dims=True))


			# self.fushion = tf.multiply(input,alpha)

			new_hidden = tf.multiply(input,alpha)


			self.atten_b_input = new_hidden
	def crf_layer(self):
		log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.decoder_targets,self.encoder_inputs_actual_length)
		self.crf_loss = tf.reduce_mean(-log_likelihood)
		self.total_loss = self.crf_loss
		#lr = tf.train.exponential_decay(self.learning_rate,
		#	                                           global_step=self.global_step,
														#    decay_steps=500,decay_rate=0.9)

		#optimizer =tf.train.GradientDescentOptimizer()
		optimizer =tf.train.AdamOptimizer(self.learning_rate)
		self.grads, self.vars = zip(*optimizer.compute_gradients(self.total_loss))
		self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradien

		self.tran_op = optimizer.apply_gradients(zip(self.gradients, self.vars))


	def step(self, sess, mode, trarin_batch,step=0):
		""" perform each batch"""
		if mode not in ['train', 'test']:
			print >> sys.stderr, 'mode is not supported'
			sys.exit(1)
		unziped = list(zip(*trarin_batch))

		if mode == 'train':
			output_feeds = [self.crf_loss,self.logits,self.transition_params,self.tran_op]
			feed_dict = {self.encoder_inputs: unziped[0],
							 self.encoder_inputs_actual_length: unziped[1],
							 self.decoder_targets: unziped[2],
							 self.global_step: step,
							self.dropout:0.5}

		else:
			output_feeds = [self.logits, self.transition_params]
			feed_dict = {self.encoder_inputs: unziped[0],
						 self.encoder_inputs_actual_length: unziped[1],

						}
		results = sess.run(output_feeds, feed_dict=feed_dict)
		return results
