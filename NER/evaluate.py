#coding:utf-8
import os
import sys
folder_name = os.path.dirname(os.path.abspath(__file__))
def eval_res(filename,res,remove=True):
	writer = open(filename,'a+',encoding='utf-8')
	writer.write(res)
	writer.close()
	if os.path.exists(filename):
		cmd = folder_name+'/conlleval.pl < %s | grep accuracy' % filename
		out = os.popen(cmd).read().split()
		if remove: os.system('rm %s' % filename)
		return {'Precision': float(out[3][:-2]), 'Recall': float(out[5][:-2]), 'F1': float(out[7]),'acc':float(out[1][:-2])}


