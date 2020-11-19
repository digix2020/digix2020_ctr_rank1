# -*- encoding: utf-8 -*-
"""
@File    : win5_groupby.py
@Time    : 2020/9/29 16:35
@Author  : lvll
@Email   : 464539082@qq.com
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from reduce.reduce import reduce, reduce_s
from gensim.models import Word2Vec
import logging
import lightgbm as lgb
from multiprocessing import Pool
import networkx as nx


# groupby-sum
def time_groupby(df, win, key, target):
    tmp = df.groupby([key, 'pt_d'], as_index=False)[target].agg({
        key + '_' + target + '_win_nunique': 'nunique',
    }).reset_index().drop('index', axis=1)
    tmp = tmp.set_index([key, "pt_d"])[[key + '_' + target + '_win_nunique']].unstack(level=-1).fillna(0)
    tmp.columns = tmp.columns.get_level_values(1)
    tmp = tmp.T
    tmp = tmp.rolling(window=win).sum().fillna(0)
    tmp = tmp.stack().reset_index()
    tmp.columns = ['pt_d', key, key + '_' + target + '_win_nunique']

    #     1,2天的sum/mean==第三天的
    tmp12 = tmp[tmp['pt_d'] < win].copy()
    del tmp12[key + '_' + target + '_win_nunique']
    tmp3 = tmp[tmp['pt_d'] == win].copy()
    del tmp3['pt_d']
    tmp12 = tmp12.merge(tmp3, on=key, how='left')
    tmp7 = tmp[tmp['pt_d'] > (win - 1)].copy()
    tmp = pd.concat([tmp12, tmp7])

    #     if tmp['win3_slide_'+fea+'sum'].var()>1:

    #     df=reduce(df)
    return tmp

def emb(df, f1, f2):
	emb_size = 8
	print('====================================== {} {} ======================================'.format(f1, f2))
	tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
	sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
	del tmp['{}_{}_list'.format(f1, f2)]
	for i in range(len(sentences)):
		sentences[i] = [str(x) for x in sentences[i]]
	model = Word2Vec(sentences, size=emb_size, window=6, min_count=5, sg=0, hs=0, seed=1, iter=5)
	emb_matrix = []
	for seq in sentences:
		vec = []
		for w in seq:
			if w in model.wv.vocab:
				vec.append(model.wv[w])
		if len(vec) > 0:
			emb_matrix.append(np.mean(vec, axis=0))
		else:
			emb_matrix.append([0] * emb_size)
	emb_matrix = np.array(emb_matrix)
	for i in range(emb_size):
		tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
	del model, emb_matrix, sentences
#     tmp = reduce(tmp)
	return tmp

##第9天数据与前7天相差太大 对数据分布进行调整
def adjust(df, key, feature):
	if key == 'uid':
		mean7 = df[df['pt_d'] < 8][feature].mean()
		std7 = df[df['pt_d'] < 8][feature].std()
		mean8 = df[(df['pt_d'] == 9) & (df['coldu'] == 1)][feature].mean()
		std8 = df[(df['pt_d'] == 9) & (df['coldu'] == 1)][feature].std()
		df.loc[(df['pt_d'] == 9) & (df['coldu'] == 1), feature]= (df[(df['pt_d'] == 9) & (df['coldu'] == 1)][feature] - mean8) / std8 * std7 + mean7
	elif key == 'adv_id' or key == 'task_id':
		mean7 = df[df['pt_d'] < 8][feature].mean()
		std7 = df[df['pt_d'] < 8][feature].std()
		mean8 = df[(df['pt_d'] == 9) & (df['colda'] == 1)][feature].mean()
		std8 = df[(df['pt_d'] == 9) & (df['colda'] == 1)][feature].std()
		df.loc[(df['pt_d'] == 9) & (df['colda'] == 1), feature]= (df[(df['pt_d'] == 9) & (df['colda'] == 1)][feature] - mean8) / std8 * std7 + mean7
	return df


def make_feature(df):
    cate_cols = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'slot_id',
                 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'city', 'city_rank', 'device_name',
                 'device_size', 'career', 'gender', 'net_type', 'residence', 'app_score', 'emui_dev',
                 'consume_purchase', 'indu_name']

    # 滑窗groupy
    nunique_group = []
    key = 'uid'
    feature_target = ['task_id', 'adv_id', 'dev_id', 'indu_name', 'adv_prim_id', 'slot_id', 'spread_app_id']
    for target in tqdm(feature_target):
        if key + '_' + target + '_win_nunique' not in nunique_group:
            nunique_group.append(key + '_' + target + '_nunique')
            tmp = time_groupby(df, 5, key, target)
            df = df.merge(tmp, on=[key, 'pt_d'], how='left')
            df = adjust(df, key, key + '_' + target + '_win_nunique')
        if target + '_' + key + '_win_nunique' not in nunique_group:
            nunique_group.append(target + '_' + key + '_win_nunique')
            tmp = time_groupby(df, 5, target, key)
            df = df.merge(tmp, on=[target, 'pt_d'], how='left')
            df = adjust(df, target, target + '_' + key + '_win_nunique')
    df = reduce(df)

    #  embedding特征
    emb_cols = [['uid', 'adv_id']]
    sort_df = df.sort_values('pt_d').reset_index(drop=True)
    for f1, f2 in emb_cols:
        df = df.merge(emb(sort_df, f1, f2), on=f1, how='left')

    #  ctr特征
    feature_list = cate_cols
    for feat_1 in tqdm(feature_list):
        res = pd.DataFrame()
        for period in range(1, 10):
            if period == 1:
                count = df[df['pt_d'] <= period].groupby(feat_1, as_index=False)['label'].agg(
                    {feat_1 + '_rate': 'mean'})
            elif period == 9:
                count = df[df['pt_d'] < 8].groupby(feat_1, as_index=False)['label'].agg({feat_1 + '_rate': 'mean'})
            else:
                count = df[df['pt_d'] < period].groupby(feat_1, as_index=False)['label'].agg({feat_1 + '_rate': 'mean'})
            count['pt_d'] = period
            res = res.append(count, ignore_index=True)
        df = pd.merge(df, res, how='left', on=[feat_1, 'pt_d'], sort=False)
        df[feat_1 + '_rate'] = reduce_s(df[feat_1 + '_rate'].fillna(-1))
        print(feat_1, ' over')

    return df


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

    df_raw = pd.read_pickle('./DIGIX_data/train_data.pkl')
    df0 = df_raw[df_raw['label'] == 0].copy()
    df1 = df_raw[df_raw['label'] == 1].copy()

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'boost_from_average': True,
        'train_metric': True,
        'feature_fraction_seed': 1,
        'learning_rate': 0.05,
        'is_unbalance': False,  # 当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
        'num_leaves': 256,  # 一般设为少于2^(max_depth)
        'max_depth': -1,  # 最大的树深，设为-1时表示不限制树的深度
        'min_child_samples': 15,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
        'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
        'subsample': 1,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 2.99,  # L1 regularization term on weights
        'reg_lambda': 1.9,  # L2 regularization term on weights
        'nthread': 12,
        'verbose': 0,
    }
    # gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=(lgb_eval), early_stopping_rounds=50)

    # epoch = gbm.best_iteration
    epoch = 550

    fold = 6
    preds = 0
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=1)
    for i, (trn_idx, val_idx) in enumerate(skf.split(df0, df0['pt_d'])):
        # if i > 0:
        # 	break

        print('fold:{}'.format(i + 1))

        df = pd.concat([df0.iloc[val_idx].reset_index(drop=True), df1], axis=0).reset_index(drop=True)
        test_df = pd.read_pickle('./DIGIX_data/test_data_B.pkl')
        test_A = pd.read_pickle('./DIGIX_data/test_data_A.pkl')
        train_uid = set(df_raw['uid'])
        test_df['coldu'] = test_df['uid'].apply(lambda x: 1 if x not in train_uid else 0)
        train_adv = set(df_raw['adv_id'])
        test_df['colda'] = test_df['adv_id'].apply(lambda x: 1 if x not in train_adv else 0)

        df = pd.concat([df, test_df], axis=0).reset_index(drop=True)
        df = pd.concat([df, test_A], axis=0).reset_index(drop=True)

        df = make_feature(df)

        test_df = df[df["pt_d"] == 9].copy().reset_index(drop=True)
        train_df = df[df["pt_d"] < 8].reset_index(drop=True)

        X_train = train_df
        y_train = X_train["label"].astype('int32')
        drop_fea = ['index', 'id', 'pt_d', 'coldu', 'colda', 'coldp', 'colds', 'label', 'communication_onlinerate']
        feature = [x for x in X_train.columns if x not in drop_fea]
        print(feature)
        weight = X_train['pt_d'] / X_train['pt_d'].max()
        # weight = X_train['pt_d'].apply(lambda x: 4 if x == 1 else x) / X_train['pt_d'].max()
        lgb_train = lgb.Dataset(X_train[feature], y_train, weight=weight)

        gbm = lgb.train(params, lgb_train, num_boost_round=epoch, valid_sets=(lgb_train))

        preds += gbm.predict(test_df[feature], num_iteration=gbm.best_iteration) / fold
        # 保存结果
        res = pd.DataFrame()
        res['id'] = test_df['id'].astype('int32')
        res['probability'] = preds
        res.to_csv('result/submission_6fold_win5.csv', index=False)