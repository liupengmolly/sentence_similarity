import os
import sys
import pickle
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
import pandas as pd
import  numpy as np
from numpy.linalg import norm
from sklearn.decomposition import NMF,PCA
from gensim import corpora,similarities,models

def cosine_array(indices_a, indices_b, values_a, values_b):
    result = 0.0
    i_a, i_b, len_a, len_b = 0, 0, len(indices_a), len(indices_b)
    while i_a < len_a and i_b < len_b:
        ind_a, ind_b = indices_a[i_a], indices_b[i_b]
        if ind_a == ind_b:
            result += values_a[i_a] * values_b[i_b]
            i_a += 1
            i_b += 1
        elif ind_a > ind_b:
            i_b += 1
        else:
            i_a += 1
    result = result / (norm(values_a) * norm(values_b))
    return result

def get_tfidf_lda_lsa_features(data):
    wcs=data[['words_chars1','words_chars2']]
    wcs['words_chars1']=wcs.words_chars1.str.split()
    wcs['words_chars2']=wcs.words_chars2.str.split()
    texts=list(np.array(wcs[['words_chars1']]))
    texts.extend(list(np.array(wcs[['words_chars1']])))
    texts=[list(x)[0] for x in texts]
    dic=corpora.Dictionary(texts)
    corpus=[dic.doc2bow(text) for text in texts]
    tfidf_model=models.TfidfModel(corpus)
    corpus_tfidf=tfidf_model[corpus]
    lda=models.LdaModel(corpus_tfidf,id2word=dic,num_topics=100)
    lsi=models.LsiModel(corpus_tfidf,id2word=dic,num_topics=100)
    sims1,sims2,sims3=[],[],[]
    for q1,q2 in np.array(wcs):
        q1_ids, q2_ids=dic.doc2bow(q1), dic.doc2bow(q2)
        tf1, tf2 = tfidf_model[q1_ids], tfidf_model[q2_ids]
        lda1,lda2 = lda[q1_ids], lda[q2_ids]
        lsi1,lsi2 = lsi[q1_ids], lsi[q2_ids]
        tk1,tv1=zip(*tf1)
        tk2,tv2=zip(*tf2)
        sims1.append(cosine_array(tk1,tk2,tv1,tv2))
        dk1,dv1=zip(*lda1)
        dk2,dv2=zip(*lda2)
        sims2.append(cosine_array(dk1,dk2,dv1,dv2))
        sk1,sv1=zip(*lsi1)
        sk2,sv2=zip(*lsi2)
        sims3.append(cosine_array(sk1,sk2,sv1,sv2))
    data['tfidf_sims']=pd.Series(sims1)
    data['lda_sims']=pd.Series(sims2)
    data['lsi_sims']=pd.Series(sims3)
    return data

def get_lda_lsi_rep_features(data,name):
    name_map={'w':'words','c':'chars','wc':'words_chars'}
    pairs = data[[name_map[name]+'1',name_map[name]+'2']]
    pairs[name_map[name]+'1'] = pairs[name_map[name]+'1'].str.split()
    pairs[name_map[name]+'2'] = pairs[name_map[name]+'2'].str.split()
    dic = pickle.load(open(os.path.join(root_path,'data_test/pai/data/features/'
                                                  'tfidf_{}_dict.pickle'.format(name)),'r'))
    corpus_tfidf = pickle.load(open(os.path.join(root_path,'data_test/pai/data/features/tfidf_{}_'
                                                           'corpustfidf.pickle'.format(name)),'r'))
    lda = models.LdaModel(corpus_tfidf, id2word=dic, num_topics=50)
    lsi = models.LsiModel(corpus_tfidf, id2word=dic, num_topics=50)
    ldas1, ldas2, lsis1, lsis2 = [], [], [], []
    for q1, q2 in np.array(pairs):
        q1_ids, q2_ids = dic.doc2bow(q1), dic.doc2bow(q2)
        lda1, lda2 = lda[q1_ids], lda[q2_ids]
        dk1, dv1 = zip(*lda1)
        dk2, dv2 = zip(*lda2)
        darray1,darray2 = np.zeros(50), np.zeros(50)
        darray1[list(dk1)] = list(dv1)
        darray2[list(dk2)] = list(dv2)
        ldas1.append(darray1)
        ldas2.append(darray2)

        lsi1, lsi2 = lsi[q1_ids], lsi[q2_ids]
        sk1, sv1 = zip(*lsi1)
        sk2, sv2 = zip(*lsi2)
        sarray1,sarray2 = np.zeros(50), np.zeros(50)
        sarray1[list(sk1)] = list(sv1)
        sarray2[list(sk2)] = list(sv2)
        lsis1.append(sarray1)
        lsis2.append(sarray2)
    data['{}ldas1'.format(name)] = pd.Series(ldas1)
    data['{}ldas2'.format(name)] = pd.Series(ldas2)
    data['{}lsis1'.format(name)] = pd.Series(lsis1)
    data['{}lsis2'.format(name)] = pd.Series(lsis2)
    return data

def get_NMF(data,emb):
    question1_vectors = np.array([x[0] for x in emb])
    question2_vectors = np.array([x[1] for x in emb])
    model=NMF(n_components=2,init='random',random_state=0)
    nmf1,nmf2=model.fit_transform(question1_vectors),model.fit_transform(question2_vectors)
    df1, df2 = pd.DataFrame(nmf1,columns=['nmf1_0','nmf1_1']), \
               pd.DataFrame(nmf2,columns=['nmf2_0','nmf2_1'])
    data = pd.concat([data,df1,df2],axis=1)
    return data

def get_PCA(data,emb):
    question1_vectors = np.array([x[0] for x in emb])
    question2_vectors = np.array([x[1] for x in emb])
    model = PCA(n_components=2)
    pca1, pca2 = model.fit_transform(question1_vectors), model.fit_transform(question2_vectors)
    df1, df2 = pd.DataFrame(pca1,columns=['pca1_0','pca1_1']), \
               pd.DataFrame(pca2,columns=['pca2_0','pca2_1'])
    data = pd.concat([data,df1,df2],axis=1)
    return data