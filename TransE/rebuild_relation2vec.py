# -*- coding:utf-8 -*-
import numpy as np
import codecs
'''
	用h+r求t的效果很差，因为学到的r表示能力太弱了，考虑用下面的方法对r向量进行重构。
	方法：对同一个r，在训练集中用t-h求和的均值作为r向量。

	@chenbingjin 2016-05-07
'''
entity2id = {}
id2entity = {}
relation2id = {}
id2relation = {}
entity2vec = {}
relation2vec = {}
relationsets = {}   # 关系集，每个r对应一个实体二元组列表[(h,t),(h`,t`)...]
n_dim = 50  		# 向量维数
d_type = np.float32	# 向量数据类型 float32/float64

def init_entity_id():
	print "Loading entity2id ..."
	with codecs.open('./data/entity2id.txt','r',encoding='utf-8') as file:
		for line in file:
			e2d = line.strip().split('\t')
			key = e2d[0]
			eid = int(e2d[1])
			entity2id[key] = eid
			id2entity[eid] = key

def init_relation_id():
	print "Loading relation2id ..."
	with codecs.open('./data/relation2id.txt','r',encoding='utf-8') as f:
		for line in f:
			r2d = line.strip().split('\t')
			rel = r2d[0]
			rid = int(r2d[1])
			relation2id[rel] = rid
			id2relation[rid] = rel

def init_entity_vector():
	print "Loading entity2vec ..."
	eid = 0
	with open('./vec/entity2vec.bern') as f:
		for line in f:
			vv = line.strip().split('\t')
			entity2vec[eid] = np.array(vv,dtype=d_type)
			eid += 1

# 数据准备
def prepare():
	print "Data Preparing ..."
	init_entity_id()
	init_relation_id()
	init_entity_vector()

def save(filename, final):
	arr = np.zeros((len(relation2id),50))
	for rid in final:
		arr[rid[0]] = rid[1]
	np.savetxt(filename, arr, fmt='%.6f', delimiter='\t')

# 重构r向量
def run():
	print "relation2id len:",len(relation2id)
	print "Build Relation sets ..."
	with codecs.open('./data/train.txt','r',encoding='utf-8') as f:
		for line in f:
			triplet = line.strip().split('\t')
			if len(triplet) != 3:
				continue
			# 获得实体和关系的id
			h = entity2id[triplet[0]]
			r = relation2id[triplet[1]]
			t = entity2id[triplet[2]]
			# 关系集
			if r not in relationsets:
				relationsets[r] = []
			relationsets[r].append((h,t))

	print "relationsets len:", len(relationsets)
	# t-h 加和平均得到r向量
	for rel in relationsets:
		rel_vec = np.zeros((1,n_dim),dtype=d_type) # 初始化0向量
		for tup in relationsets[rel]:
			t = tup[1]
			h = tup[0]
			rel_vec += (entity2vec[t]-entity2vec[h])
		rel_vec = rel_vec/len(relationsets[rel])
		relation2vec[rel] = rel_vec

	final = sorted(relation2vec.iteritems(), key=lambda x:x[0], reverse=False)
	print final[:2]

	save('./vec/relation2vec.new', final)


if __name__ == '__main__':
	prepare()
	run()
