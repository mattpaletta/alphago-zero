import os
from threading import Lock

import tensorflow as tf
import logging
import numpy as np

from config import Config


class NNet(object):
	def __init__(self,
	             board_size,
	             action_size,
	             learning_rate=0.001,
	             dropout_rate=0.3,
	             epochs=10,
	             batch_size=32,
	             num_channels=256,
	             log_device_placement=False): # TODO:// set batch_size=64, num_channels=512
		self.BOARD_SIZE_X = board_size
		self.BOARD_SIZE_Y = board_size
		self.action_size = action_size
		self.num_epochs = epochs
		self.batch_size = batch_size
		self.dropout_rate = dropout_rate
		
		# Global lock for the predict function.
		self.lock = Lock()
		
		self.graph = self.__build_model(board_size,
		                                board_size,
		                                learning_rate=learning_rate,
		                                num_channels=num_channels,
		                                action_size=action_size)

		config = tf.ConfigProto(log_device_placement=log_device_placement)
		config.gpu_options.allocator_type = 'BFC'
		#config.gpu_options.per_process_gpu_memory_fraction = 0.4
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(graph=self.graph, config=config)
		self.saver = None
		with tf.Session() as temp_sess:
			temp_sess.run(tf.global_variables_initializer())
		#self.sess.run(tf.global_variables_initializer())
		self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))
	
	def __build_model(self, board_size_x,
	                  board_size_y,
	                  learning_rate,
	                  num_channels,
	                  action_size):
		graph = tf.Graph()
		with graph.as_default():
			self.input_boards = tf.placeholder(tf.float32,
			                                   shape=[None, board_size_x, board_size_x, 1])
			self.dropout = tf.placeholder(tf.float32, name="dropout")
			self.isTraining = tf.placeholder(tf.bool, name="is_training")
			
			x_image = tf.reshape(self.input_boards, [-1, board_size_x, board_size_y, 1])
			h_conv1 = tf.nn.relu(tf.layers.batch_normalization(
					tf.layers.conv2d(x_image, num_channels, kernel_size=[3, 3], padding='same'), axis=3,
					training=self.isTraining))  # batch_size  x board_x x board_y x num_channels
			h_conv2 = tf.nn.relu(tf.layers.batch_normalization(
					tf.layers.conv2d(h_conv1, num_channels, kernel_size=[3, 3], padding='same'), axis=3,
					training=self.isTraining))  # batch_size  x board_x x board_y x num_channels
			h_conv3 = tf.nn.relu(tf.layers.batch_normalization(
					tf.layers.conv2d(h_conv2, num_channels, kernel_size=[3, 3], padding='valid'), axis=3,
					training=self.isTraining))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
			h_conv4 = tf.nn.relu(tf.layers.batch_normalization(
					tf.layers.conv2d(h_conv3, num_channels, kernel_size=[3, 3], padding='valid'), axis=3,
					training=self.isTraining))  # batch_size  x (board_x-4) x (board_y-4) x num_channels
			h_conv4_flat = tf.reshape(h_conv4, [-1, num_channels * (board_size_x - 4) * (board_size_y - 4)])
			
			s_fc1 = tf.layers.dropout(
					inputs=tf.nn.relu(
							tf.layers.batch_normalization(tf.layers.dense(h_conv4_flat, 1024), axis=1, training=self.isTraining)
					),
					rate=self.dropout)  # batch_size x 1024
			s_fc2 = tf.layers.dropout(
					inputs=tf.nn.relu(
							tf.layers.batch_normalization(tf.layers.dense(s_fc1, 512), axis=1, training=self.isTraining)),
					rate=self.dropout)  # batch_size x 512

			pi = tf.layers.dense(s_fc2, action_size)  # batch_size x self.action_size
			self.prob = tf.nn.softmax(pi)
			self.value = tf.nn.tanh(tf.layers.dense(s_fc2, 1))  # batch_size x 1
			
			self.__calculate_loss(learning_rate=learning_rate,
			                      action_size=action_size,
			                      value=self.value,
			                      prob=self.prob)
		summary = tf.summary.FileWriter("logs/")
		summary.add_graph(graph)
		return graph
	
	def __calculate_loss(self, learning_rate, action_size, value, prob):
		self.target_pis = tf.placeholder(tf.float32, shape=[None, action_size])
		self.target_vs = tf.placeholder(tf.float32, shape=[None])
		self.loss_pi = tf.losses.softmax_cross_entropy(self.target_pis, prob)
		self.loss_v = tf.losses.mean_squared_error(self.target_vs, tf.reshape(value, shape=[-1, ]))
		total_loss = self.loss_pi + self.loss_v
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
	
	def train(self, examples):
		for epoch in range(self.num_epochs):
			num_batches = int(len(examples) / self.batch_size)
			for batch_idx in range(num_batches):
				sample_ids = np.random.randint(len(examples), size=self.batch_size)
				boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
				
				# Reshape so tensorflow is happy.
				num_boards = len(boards)
				board_size = Config().get_args().board_size
				boards = np.asarray(list(boards)).reshape((num_boards, board_size, board_size, 1))
				
				# predict and compute gradient and do SDG step
				input_dict = {
					self.input_boards: boards,
					self.target_pis:   list(pis),
					self.target_vs:    list(vs),
					self.isTraining:   True,
					self.dropout:      self.dropout_rate
				}
				
				res = self.sess.run(
						fetches={
							"train_step": self.train_step,
							"loss_pi":    self.loss_pi,
							"loss_v":     self.loss_v
						},
						feed_dict=input_dict
				)
				logging.info("({0}/{1}:{2}/{3}) PI: {4:03f} V: {5:03f}".format(batch_idx+1,
				                                                             num_batches,
				                                                             epoch+1,
				                                                             self.num_epochs,
				                                                             res["loss_pi"],
				                                                             res["loss_v"]))
	
	def predict(self, board):
		# preparing input
		board = board[np.newaxis, :, :]
		self.lock.acquire()
		res = self.sess.run(
			fetches={
				"prob":  self.prob,
				"value": self.value
			},
			feed_dict={
				self.input_boards:  board,
				self.dropout:       0,
				self.isTraining:    False
			}
		)
		
		self.lock.release()
		
		prob = res["prob"]
		v = res["value"]
		return prob[0], v[0]
	
	def save_checkpoint(self, folder, filename):
		save_file_path = os.path.join(folder, filename)
		if not os.path.exists(folder):
			logging.info("Checkpoint directory does not exist, making directory: " + str(folder))
			os.mkdir(folder)
		else:
			logging.info("Checkpoint directory exists: " + str(folder))
		
		if self.saver is None:
			logging.debug("Creating new saver")
			self.saver = tf.train.Saver(self.graph.get_collection_ref("variables"))
		with self.graph.as_default():
			logging.info("Saving model to: " + str(save_file_path))
			self.saver.save(self.sess, save_file_path)
			logging.info("Saved model to: " + str(save_file_path))
	
	def load_checkpoint(self, folder, filename):
		load_file_path = os.path.join(folder, filename)
		logging.debug("Loading model from : " + str(load_file_path))
		if not os.path.exists(load_file_path + ".meta"):
			raise ("No model found in path: " + str(load_file_path))
		with self.graph.as_default():
			logging.info("Loaded saved model.")
			self.saver = tf.train.Saver()
			self.saver.restore(self.sess, load_file_path)
