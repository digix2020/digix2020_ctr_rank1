import pandas as pd
import numpy as np
from tqdm import tqdm

def reduce(df):
	int_list = ['int', 'int32', 'int16']
	float_list = ['float', 'float32']
	for col in tqdm(df.columns):
		col_type = df[col].dtypes
		if col_type in int_list:
			c_min = df[col].min()
			c_max = df[col].max()
			if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
				df[col] = df[col].astype(np.int8)
			elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
				df[col] = df[col].astype(np.int16)
			elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
				df[col] = df[col].astype(np.int32)
		elif col_type in float_list:
			c_min = df[col].min()
			c_max = df[col].max()
			if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
				df[col] = df[col].astype(np.float16)
			elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
				df[col] = df[col].astype(np.float32)
	return df

def reduce_s(df):
	int_list = ['int', 'int32', 'int16']
	float_list = ['float', 'float32']
	col_type = df.dtypes
	if col_type in int_list:
		c_min = df.min()
		c_max = df.max()
		if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
			df = df.astype(np.int8)
		elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
			df = df.astype(np.int16)
		elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
			df = df.astype(np.int32)
	elif col_type in float_list:
		c_min = df.min()
		c_max = df.max()
		if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
			df = df.astype(np.float16)
		elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
			df = df.astype(np.float32)
	return df

if __name__ == "__main__":
	print('开始压缩训练集')
	df = pd.read_csv('../data/train_data.csv',sep='|')
	df = reduce(df)
	df.to_pickle('../data/train_data.pkl')
	print('训练集压缩完成，开始压缩测试集')
	df = pd.read_csv('../data/test_data_A.csv',sep='|')
	df = reduce(df)
	df.to_pickle('../data/test_data_A.pkl')
	df = pd.read_csv('../data/test_data_B.csv',sep='|')
	df = reduce(df)
	df.to_pickle('../data/test_data_B.pkl')
	print('测试集压缩完成')
