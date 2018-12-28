#coding:utf-8
#coding:utf-8 
from data_loader import *
from bilstm_crf import *
import sys
from evaluate import *

import numpy as np
np.random.seed(7)
import tensorflow as tf
tf.set_random_seed(7)

input_steps = 40
embedding_size = 300
hidden_size = 100
n_layers = 1
batch_size =20
vocab_size =4002
slot_size = 11
#vocab_size =1824
#slot_size = 26
#batch_size =16

epoch_num = 100



#embedding_file = ''
embedding_file = '/search/folder/glove/glove.840B.300d.txt'
#embedding_file = '/search/folder/glove/glove.6B.100d.txt'
#embedding_file = '/search/folder/music/pretrain_music_emb.vec'
def load_embedding(word_index,EMBEDDING_FILE):
	def get_coefs(word, *arr):
		if len(arr)!=300:return word, np.array([0]*300,dtype='float32')#None #print(word+' '+str(len(arr))) #return word, np.asarray([float(d) for d in arr], dtype='float32')
		return word, np.asarray([float(d) for d in arr], dtype='float32')

	embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE,encoding='utf-8'))

	all_embs = np.stack(embeddings_index.values())
	emb_mean, emb_std = all_embs.mean(), all_embs.std()
	embed_size = all_embs.shape[1]

	# word_index = tokenizer.word_index
	oov=0# word_index = tokenizer.word_index
	nb_words = len(word_index)
	embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
	for word, i in word_index.items():
		# if i >= max_features: continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None: embedding_matrix[i] = embedding_vector
		else: oov+=1#print(word)#oov+=1#oov+=1#if embedding_vector is not None: embedding_matrix[i] = embedding_vector
	print(float(oov/nb_words))#=0# word_index = tokenizer.word_index

	return embedding_matrix

def get_model(embedding_matrix):
	model = Model(input_steps, embedding_size, hidden_size, vocab_size, slot_size, batch_size, n_layers,embedding_matrix)
	model.build_graph()
	return model


def train():
	vocab_word2index, vocab_idx2word, tag2index, idx2tag = build_vocab('train.txt')
	index_train = to_index('train.txt', vocab_word2index, tag2index,input_steps)
	index_dev = to_index('dev.txt', vocab_word2index, tag2index,input_steps)
	index_test = to_index('test.txt', vocab_word2index, tag2index,input_steps)
	if embedding_file!='':
		embedding_matrix = load_embedding(vocab_word2index,embedding_file)
	else:
		embedding_matrix = None


	model = get_model(embedding_matrix)

	sess = tf.Session()

	sess.run(tf.global_variables_initializer())

	print(len(vocab_word2index))
	print(len(tag2index))

	step = 0
	best_f1 = 0.0
	early_stop_patient = 0
	best_acc = 0.0
	best_test_acc = 0.0
	for epoch in range(epoch_num):
		for i, batch in enumerate(getBatch(batch_size, index_train,True)):
			# 执行一个batch的训练
			unzip = list(zip(*batch))
			seq_lens = unzip[1]
			#print([vocab_idx2word[index] for index in  unzip[0][0]])
			#print([idx2tag [index] for index in  unzip[2][0]])
			#sys.exit(1)
			target_input = unzip[2]
			loss, logits, transition_params, _ = model.step(sess, "train", batch,step)
			predicted_ids = []
			for logit, length in zip(logits, seq_lens):
				logit = logit[:length]
				predicted_id, score = tf.contrib.crf.viterbi_decode(logit, transition_params)
				predicted_ids.append(predicted_id)

			mask = (np.expand_dims(np.arange(input_steps), axis=0) < np.expand_dims(seq_lens, axis=1))
			correct_count = 0
			for idx in range(len(predicted_ids)):
				for j in range(len(predicted_ids[idx])):
					if mask[idx][j] and predicted_ids[idx][j] == target_input[idx][j]:
						correct_count += 1
			total = np.sum(mask)
			correct_labels = np.sum((target_input == np.array(predicted_ids)) * mask)

			if (step + 1) % 100 == 0:
				print('epoch: (%d / %d) | loss : %.4f |  acc : %.4f ' % (epoch + 1, epoch_num, loss, float(correct_count) / total))
			step += 1

			if  (step+1) % 500 == 0 :
				res = ''
				valid_acc =0.0
				right = 0.0
				total =0.0
				for index, test_batch in enumerate(getBatch(batch_size, index_dev)):
					unzip = list(zip(*test_batch))

					test_seq_lens = unzip[1]
					test_input = list(unzip[2])
					input_text = unzip[0]
					#print(input_text = unzip[0]
					test_predict = []
					logits, transition_params= model.step(sess, "test", test_batch)
					for logit, length in zip(logits, test_seq_lens):
						logit = logit[:length]
						predicted_id, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
						test_predict.append(predicted_id)
					for idx in range(len(test_predict)):
						out = 'BOS O O\n'
						for j in range(len(test_predict[idx])):
							try:
								out += vocab_idx2word[input_text[idx][j]] + ' ' + idx2tag[
									test_input[idx][j]] + ' ' + idx2tag[test_predict[idx][j]] + '\n'
							except Exception as e:
								print("pred....." + str(test_predict[i][j]))
								print("text_Input..." )
								print(len(input_text))
								print(idx)
								print(test_input)
								print(len(input_text[idx]))
								print("string ...." + str(j))
								print("input_text...." + str(input_text[i][j]))
								sys.exit(1)

						out += 'EOS O O \n\n'
						res += out

				filename = 'validate_tmp.txt'
				eval_str = eval_res(filename, res,False)
				json_data = eval_str
				if json_data['F1'] > best_f1:
					best_f1 = json_data['F1']
					print(eval_str)
					import os #best_f1 = json_data['F1']
					os.system('rm %s' % filename)#print(eval_str)
					test(model, index_test, sess, vocab_idx2word, idx2tag)
					# print(" valid acc : "+str(right/total))
					early_stop_patient = 0
				else:
					early_stop_patient += 1
				if early_stop_patient > 15:
					print("best f1" + str(best_f1))
					sys.exit(1)



def test(model,index_dev,sess,index2word,index2slot):
	res = ''
	for j, test_batch in enumerate(getBatch(batch_size, index_dev)):
		logits, transition_params = model.step(sess, "test", test_batch)
		unzip = list(zip(*test_batch))
		test_seq_lens = unzip[1]
		test_input = unzip[2]
		input_text = unzip[0]
		test_predict = []
		for logit, length in zip(logits, test_seq_lens):
			logit = logit[:length]
			predicted_id, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
			test_predict.append(predicted_id)

		for i in range(len(test_predict)):
			out = 'BOS O O\n'
			for jIndex in range(len(test_predict[i])):
				out += index2word[input_text[i][jIndex]] + ' ' + index2slot[
					test_input[i][jIndex]] + ' ' + index2slot[test_predict[i][jIndex]] + '\n'

			out += 'EOS O O \n\n'
			res += out
	filename = 'test_tmp.txt'
	eval_str = eval_res(filename, res)
	print("...................test.................")
	print(eval_str)
	print("...................test.................")


if __name__=='__main__':
	train()
