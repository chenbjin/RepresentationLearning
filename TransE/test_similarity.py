# -*- coding: utf-8 -*-
import codecs
import numpy as np
from scipy.spatial.distance import cosine
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
'''
	find topK similar entities.
	@chenbingjin 2016-05-06
'''
entity2id = {}
id2entity = {}
relation2id = {}
id2relation = {}
entity2vec = {}
relation2vec = {}

def init_entity_id():
	print "load entity2id ..."
	with codecs.open('./data/entity2id.txt','r',encoding='utf-8') as file:
		for line in file:
			e2d = line.strip().split('\t')
			key = e2d[0]
			eid = int(e2d[1])
			entity2id[key] = eid
			id2entity[eid] = key

def init_relation_id():
	print "load relation2id ..."
	with codecs.open('./data/relation2id.txt','r',encoding='utf-8') as f:
		for line in f:
			r2d = line.strip().split('\t')
			rel = r2d[0]
			rid = int(r2d[1])
			relation2id[rel] = rid
			id2relation[rid] = rel

def init_entity_vector():
	eid = 0
	print "load entity2vec ..."
	with open('./vec/entity2vec.bern') as f:
		for line in f:
			vv = line.strip().split('\t')
			entity2vec[eid] = np.array(vv,dtype=np.float32)
			eid += 1

def init_relation_vector():
	rid = 0
	print "load relation2vec ..."
	with open('./vec/relation2vec.bern') as f:
		for line in f:
			vv = line.strip().split('\t')
			relation2vec[rid] = np.array(vv,dtype=np.float32)
			rid += 1

'''
	获取与head+rel相似的实体
'''
def sim_cosine(head, rel):
	print 'finding entity (h+r=t) ...'
	ee = entity2vec[entity2id[head]]+relation2vec[relation2id[rel]]
	ans = {}
	for en in entity2id:
		eid = entity2id[en]
		evec = entity2vec[eid]
		ans[eid] = cosine(ee,evec)
	ans = sorted(ans.iteritems(),key=lambda x:x[1],reverse=False)
	return ans

def sim_relation(head, tail):
	print 'finding relation (r=t-h) ...'
	rr = entity2vec[entity2id[tail]]-entity2vec[entity2id[head]]
	ans = {}
	for rel in relation2id:
		rid = relation2id[rel]
		rvec = relation2vec[rid]
		ans[rid] = cosine(rr,rvec)
	ans = sorted(ans.iteritems(),key=lambda x:x[1],reverse=False)
	return ans
'''
	获取相似的实体
'''
def sim_entity(head):
	ee = entity2vec[entity2id[head]]
	ans = {}
	print 'similar entity finding ...'
	for en in entity2id:
		eid = entity2id[en]
		evec = entity2vec[eid]
		ans[eid] = cosine(ee,evec)
	ans = sorted(ans.iteritems(),key=lambda x:x[1],reverse=False)
	return ans

if __name__ == '__main__':
	init_entity_id()
	init_relation_id()
	init_entity_vector()
	init_relation_vector()
	print "\nThree choices: "
	print "\t0.entity similarity;\n\t1.find t (t = h+r);\n\t2.find r (r = t-h)\n"
	while True:
		x = input("input choice (0/1/2):")
		if int(x) == 0:
			entity = raw_input("input entity: ")
			sim = sim_entity(unicode(entity))
			print "-----------top30---------------"
			for e in sim[:30]:
				eid = e[0]
				print id2entity[eid]
		elif int(x) == 1:
			head = raw_input("input head entity: ")
			rel = raw_input("input relation: ")
			sim = sim_cosine(unicode(head), unicode(rel))
			print "-----------top30---------------"
			for e in sim[:30]:
				eid = e[0]
				print id2entity[eid]
		else:
			head = raw_input("input head entity: ")
			tail = raw_input("input tail entity: ")
			sim = sim_relation(unicode(head), unicode(tail))
			print "-----------top30---------------"
			for e in sim[:30]:
				rid = e[0]
				print id2relation[rid]
