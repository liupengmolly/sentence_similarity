from lib.feature.nlp import *
from lib.feature.representation import *
from lib.feature.stats import *
from lib.feature.graph import *


def get_feature(data,emb,num=0):
    ori_cols=list(data.columns)
    #stats features
    data = get_basic_feature(data)
    data = get_fuzzy_feature(data)
    data = get_lda_lsi_rep_features(data,'wc')
    data = get_distance_features(data, emb)
    data = get_stats_features(data, emb)
    data = get_pword(data)
    data = get_pword_rate(data)

    #representation features
    data=get_NMF(data,emb)
    data=get_PCA(data,emb)

    #graph features
    data = get_clique_feature(data)
    data = get_connected_feature(data)
    data = get_nodes_features(data)
    data = get_edges_features(data)
    feature_cols=list(data.columns)
    for col in ori_cols:
        feature_cols.remove(col)
    features = np.array(data[feature_cols])
    features_list=[]
    for feature in features:
        feature_list=[]
        for f in feature:
            if isinstance(f,list) or isinstance(f,np.ndarray) or isinstance(f,tuple):
                feature_list.extend(f)
            else:
                feature_list.append(f)
        features_list.append(feature_list)
    print(features_list[0])
    return np.array(features_list)


