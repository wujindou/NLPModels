#coding:utf-8

from collections import Counter
import sys
import random
import re

#folder = '/Users/apple/Downloads/AutoNER-master/data/music/'
folder = '/search/folder/dataset/conll2003/en/'
#folder = '/search/folder/dataset/NERData/CoNLL-2003/'
#folder = '/search/folder/dataset/NERData/conll_2003_bioes/'
#folder = '/search/folder/music/'
#folder = '/search/folder/music_new/'
PAD_ID = 0
UNK_ID = 1
_PAD="_PAD"
_UNK="<UNK>"
sep=' '
def build_vocab(file,vocab_size =4000):
	c_input= Counter()
	slot_tag = set()
	for file in ['train.txt']:
		with open(folder+file,'r',encoding='utf-8') as lines:
				for line in lines:
					if len(line.strip().split(sep))==1:continue
					data = line.strip().split(sep)
					words = data[0]
					words = data[0].lower()
					slot_tag.add(data[-1])
					words = re.sub('\d', '0', words)#words.isdigit():print(words) # words='0'
					c_input[words]+=1
	vocab_list = c_input.most_common()
	#vocab_list = c_input.most_common(vocab_size)
	vocab_word2index = {}
	vocab_word2index[_PAD] = PAD_ID
	vocab_word2index[_UNK] = UNK_ID
	for i,tuplee in enumerate(vocab_list):
			word,freq = tuplee
			vocab_word2index[word]=i+2
	vocab_idx2word = {v:k for k,v in  vocab_word2index.items()}
	tag2index = {}
	tag2index[_PAD] =PAD_ID
	tag2index[_UNK] =UNK_ID
	for tag in slot_tag:
		if tag not in tag2index.keys():
			tag2index[tag] = len(tag2index)
	idx2tag = {v:k for k,v in tag2index.items()}
	return vocab_word2index,vocab_idx2word,tag2index,idx2tag

def padSentence(s, max_length):
	if len(s)>max_length:
		return s[:max_length]
	else:
		return s + [PAD_ID]*(max_length - len(s))

def to_index(filename, word2index, slot2index,max_len=20):
	new_train = []
	temp_words = []
	temp_labels = []
	with open(folder+filename,'r',encoding='utf-8') as lines:
		for line in lines:
			if len(line.strip().split(sep))==1:
				sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],temp_words))
				true_length = len(temp_words)
				sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],temp_labels))

				sin_ix_padded = padSentence(sin_ix,max_len)
				sout_ix_padded = padSentence(sout_ix,max_len)


				new_train.append([sin_ix_padded, true_length, sout_ix_padded])
				temp_words =[]
				temp_labels=[]
			else:
				#if not line.strip().lower().split(sep)[0].isdigit():temp_words.append(line.strip().lower().split(sep)[0])
				#else: temp_words.append('0')
				#temp_words.append(re.sub('\d','0',line.strip().split(sep)[0]))
				temp_words.append(re.sub('\d','0',line.strip().lower().split(sep)[0]))
				temp_labels.append(line.strip().split(sep)[-1])
	return new_train

def getBatch(batch_size, train_data,shuffle=False):
    if shuffle : random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch



if __name__=='__main__':
	vocab_word2index, vocab_idx2word, tag2index, idx2tag = build_vocab('train.txt')
	to_index('train.txt',vocab_word2index,tag2index)
	print(tag2index)#to_index('train.txt',vocab_word2index,tag2index)
	print(len(vocab_word2index))#to_index('train.txt',vocab_word2index,tag2index)
	print(len(tag2index))#to_index('train.txt',vocab_word2index,tag2index)

	index_train = to_index('train.txt', vocab_word2index,tag2index)
	index_dev = to_index('dev.txt', vocab_word2index,tag2index)
	index_test = to_index('test.txt',vocab_word2index,tag2index)
