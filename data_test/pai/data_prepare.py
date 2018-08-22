# !/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import networkx as nx
import itertools

def transform_embedding(df, question):
    """
    将id对训练数据转换为embedding对
    如对于竞赛数据，test.csv->predict_data.csv
    :param df:
    :param question:
    :return:
    """
    question['wcs'] = question['words'] + ' ' + question['chars']
    qs1 = list(df['q1'])
    qs2 = list(df['q2'])
    def transform(qs):
        ws, cs, wcs = [], [], []
        for q in qs:
            idx = int(q[1:])
            w = question.loc[idx, 'words']
            c = question.loc[idx, 'chars']
            wc = question.loc[idx, 'wcs']
            ws.append(w)
            cs.append(c)
            wcs.append(wc)
        return ws, cs, wcs
    ws1, cs1, wcs1 = transform(qs1)
    ws2, cs2, wcs2 = transform(qs2)
    df['words1'] = pd.Series(ws1)
    df['chars1'] = pd.Series(cs1)
    df['words_chars1'] = pd.Series(wcs1)
    df['words2'] = pd.Series(ws2)
    df['chars2'] = pd.Series(cs2)
    df['words_chars2'] = pd.Series(wcs2)
    return df


def get_dict(df):
    """
    生成数据时，用字典记录数据的边和节点
    :param df:
    :return:
    """
    import numpy as np
    from collections import defaultdict
    q_dict = defaultdict(dict)
    data = np.array(df)
    for q1, q2, label in data:
        q_dict[q1][q2] = int(label)
        q_dict[q2][q1] = int(label)
    return q_dict


def create_data(df):
    """
    生成构造数据
    :param df:
    :return:
    """
    q_dict = get_dict(df)
    new_data = {}
    for k in q_dict:
        k_dict = q_dict[k]
        for kk in k_dict:
            kk_dict = q_dict[kk]
            if len(kk_dict) > 1:
                for kkk in kk_dict:
                    if kkk != k and ((kkk, k) not in new_data) and ((k, kkk) not in new_data):
                        label1, label2 = k_dict[kk], kk_dict[kkk]
                        if label1 != 0 or label2 != 0:
                            new_data[(k, kkk)] = (label1 and label2)
    new_data_list = []
    for k, v in new_data.items():
        new_data_list.append((k[0], k[1], v))
    return pd.DataFrame(new_data_list, columns=['q1', 'q2', 'label'])


def convert_to_txt(v, path):
    file = open(path, 'w')
    v_list = []
    for i in v:
        l = ' '.join([str(j) for j in i]) + '\n'
        v_list.append(l)
    file.writelines(v_list)


def generate_data_by_graph(data_path):
    """
    生成全连通图数据（小涛哥编写）
    :param data_path:
    :return:
    """
    train_data = pd.read_csv(data_path)
    train_data_ = zip(list(train_data.label), list(train_data.q1), list(train_data.q2))
    graph = nx.DiGraph()
    for ele in train_data_:
        if ele[0] == 1:
            graph.add_edge(ele[1], ele[2])
    sub_graph = [x for x in nx.weakly_connected_components(graph)]

    pos_edges_generated = []
    for sub in sub_graph:
        sub = list(sub)
        sub_edges = itertools.product(sub, sub)
        sub_edges = [x for x in sub_edges if x[1] != x[1]]
        pos_edges_generated.extend(sub_edges)

    df = pd.DataFrame(pos_edges_generated)
    df.to_csv('pos_edges_generated.csv')

    sub_id = [(sorted(x)[0], list(x)) for x in sub_graph]
    sub_id_ = dict(sub_id)
    node_sub_id = {}
    for item in sub_id:
        index = item[0]
        sub = item[1]
        for node in sub:
            node_sub_id[node] = index

    neg = [(y[1], y[2]) for y in train_data_ if y[0] == 0]
    neg_ids = []
    for ele in neg:
        id0 = node_sub_id.get(ele[0], ele[0])
        id1 = node_sub_id.get(ele[1], ele[1])
        neg_ids.append((id0, id1))
    neg_ids = set(neg_ids)
    neg_edges_generated = []
    for ele in neg_ids:
        graph1 = sub_id_.get(ele[0], [ele[0], ])
        graph2 = sub_id_.get(ele[1], [ele[1], ])
        pairs = itertools.product(graph1, graph2)
        neg_edges_generated.extend(pairs)

    df = pd.DataFrame(neg_edges_generated)
    df.to_csv('neg_edges_generated.csv')


def log_sample(df,num):
    """
    依据df自身的log分布，取样num条数据
    :param df:
    :param num:
    :return:
    """
    import numpy as np
    dfq1=df.groupby('q1')
    dfq2=df.groupby('q2')
    def get_result(dfq,othercol):
        dfcount=dfq.count()
        dfcount['log']=dfcount[othercol].apply(np.log)
        dfcount['lograte']=dfcount['log']/dfcount['log'].sum()
        dflist=[]
        for name,g in dfq:
            rate=dfcount.loc[name,'lograte']
            sp_num=int(rate*num/2)
            if sp_num==0:
                sp_num+=1
            sp_df=g.sample(n=sp_num)
            dflist.append(sp_df)
        result=pd.concat(dflist)
        return result
    result1=get_result(dfq1,'q2')
    result2=get_result(dfq2,'q1')
    result=pd.concat([result1,result2]).drop_duplicates()
    return result

def pai_sample(df,com,num,com_rate=0.9,no_direction=True):
    """
    从df中依据分布取样,分为包含共有句子com集合的取样和不包含共有句子com的取样，
    no_direction表示df中的句子对是否无向
    :param df:
    :param com:
    :param num:
    :param com_rate:
    :param no_direction:
    :return:
    """
    def cross_data(com, gen, no_direction=True):
        comcol = com.columns[0]
        com = com.rename(columns={comcol: 'q1'})
        comgen = pd.merge(gen, com, on='q1')
        if not no_direction:
            com = com.rename(columns={'q1': 'q2'})
            genq2 = pd.merge(gen, com, on='q2')
            comgen = pd.concat([comgen, genq2]).drop_duplicates()
        nocomgen = pd.concat([gen, comgen]).drop_duplicates(keep=False)
        return comgen, nocomgen
    def direction_drop_dup(pos):
        pos = pos.rename(columns={'0': 'q1', '1': 'q2'})
        pos['add'] = pos.q1.str[1:].astype(int) + pos.q2.str[1:].astype(int)
        pos['sub'] = pos.q1.str[1:].astype(int) - pos.q2.str[1:].astype(int)
        pos['sub'] = pos['sub'].apply(abs)
        pos = pos.drop_duplicates(['add', 'sub'])
        cols=list(pos.columns)
        cols.remove('add')
        cols.remove('sub')
        return pos[cols]
    if num > len(df):
        return ValueError('the sample num {} is larger than the dataframe size {}'
                          .format(num, len(df)))
    com_num, nocom_num = int(num*com_rate), int(num*(1-com_rate))
    comdf, nocom_df = cross_data(com, df, no_direction)
    com_result=log_sample(comdf,com_num)
    if len(nocom_df)<nocom_num:
        nocom_result=nocom_df
    else:
        nocom_result=log_sample(nocom_df,nocom_num)
    result=pd.concat([com_result,nocom_result])
    result=direction_drop_dup(result)
    return result

def log_sample_com(rate_q,sp_df,num,rate):
    """
    依据rate_q的分布取样num条sp_df中的数据
    :param rate_q:
    :param sp_df:
    :param num:
    :param rate:
    :return:
    """
    import numpy as np
    reverse_sp_df=sp_df.rename(columns={'q1':'q2','q2':'q1'})[['q1','q2','label']]
    df = pd.concat([sp_df,reverse_sp_df])
    group_df=df.groupby('q1')
    groups=[]
    rate_q['q1']=rate_q.index
    rate_q=pd.merge(rate_q,pd.DataFrame(df['q1'].unique(),columns=['q1']),on='q1')\
        .reset_index(drop=True)
    if rate=='log':
        rate_q['log'] = (rate_q['id']+1).apply(np.log)
        rate_q['rate'] = rate_q['log']/rate_q['log'].sum()
    elif rate=='num':
        rate_q['rate'] = rate_q['id'] / rate_q['id'].sum()
    else:
        return ValueError('the rate arguments must in [num,log], not {}.'.format(rate))
    for i in rate_q.index:
        q=rate_q.loc[i,'q1']
        group=group_df.get_group(q)
        sp_rate = rate_q.loc[i,'rate']*num/len(group)
        if sp_rate>1:
            groups.append(group)
        else:
            groups.append(group.sample(frac=sp_rate))
    result=pd.concat(groups).drop_duplicates().sample(frac=1).reset_index(drop=True)
    return result


def pai_sample_com(rate_df,sp_df,num,no_direction=True,rate='num'):
    """
        从sp_df中依据rate_df的分布取样，只取样rate_df和sp_df共有的句子在sp_df中出现的部分。
        no_direction表示df中的句子对是否无向
        :param df:
        :param com:
        :param num:
        :param com_rate:
        :param no_direction:
        :return:
    """
    rate_df=rate_df[['q1','q2']]
    q_df=pd.concat([rate_df[['q1']].rename(columns={'q1':'q'}),
                    rate_df[['q2']].rename(columns={'q2':'q'})])
    unique_q=pd.DataFrame(q_df['q'].unique(),columns=['q'])
    group_df=q_df
    group_df['id']=pd.Series(len(group_df)*[1])
    rate_q=group_df.groupby('q').count()
    def cross_data_com(unique_q, sp_df, no_direction=True):
        comcol = unique_q.columns[0]
        unique_q = unique_q.rename(columns={comcol: 'q1'})
        comgen = pd.merge(sp_df, unique_q, on='q1')
        if not no_direction:
            unique_q = unique_q.rename(columns={'q1': 'q2'})
            genq2 = pd.merge(sp_df, unique_q, on='q2')
            comgen = pd.concat([comgen, genq2]).drop_duplicates()
        return comgen
    def direction_drop_dup(pos):
        pos = pos.rename(columns={'0': 'q1', '1': 'q2'})
        pos['add'] = pos.q1.str[1:].astype(int) + pos.q2.str[1:].astype(int)
        pos['sub'] = pos.q1.str[1:].astype(int) - pos.q2.str[1:].astype(int)
        pos['sub'] = pos['sub'].apply(abs)
        pos = pos.drop_duplicates(['add', 'sub'])
        cols=list(pos.columns)
        cols.remove('add')
        cols.remove('sub')
        return pos[cols]
    if num > len(sp_df):
        return ValueError('the sample num {} is larger than the dataframe size {}'
                          .format(num, len(sp_df)))
    comdf = cross_data_com(unique_q, sp_df, no_direction)
    com_result=log_sample_com(rate_q,comdf,num,rate)
    result=direction_drop_dup(com_result)
    return result

def get_train_data(data,question,train_data):
    data=transform_embedding(data,question)
    total_data=pd.concat([train_data,data]).sample(frac=1).reset_index(drop=True)
    total_data.to_csv("train_data.csv")
    train1_num=int(len(total_data)*0.7)
    total_data[:train1_num].to_csv("train_data1.csv")
    total_data[train1_num:].to_csv("train_data2.csv")

def transform_to_train(df,question,label):
    df=df.reset_index(drop=True)
    df['qs']=df['0'].str.split(',')
    df['q1']=df['qs'].apply(lambda x:x[0][2:-1])
    df['q2']=df['qs'].apply(lambda x:x[1][2:-2])
    df['label']=pd.Series(len(df)*[label])
    df=transform_embedding(df[['label','q1','q2']],question)
    return df

def sample(rate_pos,rate_neg,pos,neg,num,id):
    pos_num,neg_num= int(num/2),int(num/2)
    sp_pos=pai_sample_com(rate_pos,pos,int(1.2*pos_num)).sample(pos_num)
    sp_neg=pai_sample_com(rate_neg,neg,int(1.2*neg_num)).sample(neg_num)
    sp=pd.concat([sp_pos,sp_neg]).sample(frac=1).reset_index(drop=True)
    sp.to_csv("train_5_30_test_30_ensemble_10/ens{}.csv".format(id))
    return sp

def get_stopwords(question):
    """
    依据词频找出question中的停用词和停用字
    :param question:
    :return:
    """
    words=' '.join(question[['words']].values.flat).split()
    chars=' '.join(question[['chars']].values.flat).split()
    wdic,cdic={},{}
    for word in words:
        wdic[word]=wdic.get(word,0)+1
    for char in chars:
        cdic[char]=cdic.get(char,0)+1
    return wdic,cdic

def remove_stopwords(sent,words):
    sent_words=sent.split()
    for x in sent_words:
        if x in words:
            sent_words.remove(x)
    return ' '.join(sent_words)


def judge(x):
    if x['if'] == True:
        if x['y_pre'] >= 0.5 and x['old'] < 0.5 and x['mod'] == 0:
            return 1.0
        if x['y_pre'] < 0.5 and x['old'] >= 0.5 and x['mod'] == 1:
            return 0.0
        return x['mod']
    else:
        return x['mod']

















