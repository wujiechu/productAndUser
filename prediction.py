# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

base = pd.read_csv('base_info.csv')
container = pd.read_csv('container_info.csv')

base.loc[:,'bid'] = base['id']
del base['id']

container = container.dropna()
container = container[container['service_name'] != 'NONE_NONE.0']
del container['id']

# 打印基本信息
print base.columns
base.info()

base.loc[:,'sec'] = np.array((base['bid'] - 253) / 3, dtype=np.int32) * 10
base.loc[base.node_ip=='172.16.1.90','node_id'] = '0'
base.loc[base.node_ip=='172.16.1.91','node_id'] = '1'
base.loc[base.node_ip=='172.16.1.101','node_id'] = '2'

base.loc[:,'hour'] = np.array(base['sec'] / 3600, dtype=np.int32)
base.loc[:,'min'] = np.array(base['sec'] / 60, dtype=np.int32)
base.loc[:,'mem_rate'] = base['mem_use'] / base['mem_total']
base.loc[:,'disk_rate'] = base['disk_use'] / base['disk_total']

base.loc[:,'sec'] = np.array((base['bid'] - 253) / 3, dtype=np.int32) * 10
base.loc[base.node_ip=='172.16.1.90','node_id'] = '0'
base.loc[base.node_ip=='172.16.1.91','node_id'] = '1'
base.loc[base.node_ip=='172.16.1.101','node_id'] = '2'

base.loc[:,'hour'] = np.array(base['sec'] / 3600, dtype=np.int32)
base.loc[:,'min'] = np.array(base['sec'] / 60, dtype=np.int32)
base.loc[:,'mem_rate'] = base['mem_use'] / base['mem_total']
base.loc[:,'disk_rate'] = base['disk_use'] / base['disk_total']

# 按照左连接方式，合并两个数据表
new_df = pd.merge(container, base, on=['bid'], how='left')

# 由于没有container_id，缺乏对sample的唯一标识，合并相同<service, sec>的样例
#avg_cols = ['cpu','mem','mem_use','mem_total','net_in','net_out','disk_in','disk_out']
id_cols = ['sec','service_name']
# 一个时间点只存在一个相同service_name 记录，对多个记录取平均值
new_df = new_df.groupby(id_cols).mean().reset_index()

# 查看各属性得分布情况，没变化得属性不进行特征提取
for col in new_df.columns:
    print new_df[new_df.service_name==0][col].describe()
    print '*'*10

import time

def feat_last_time(df_data, col, time_col, interval):
    # 上一个时间点的特征值
    # e.g. last_10_sec_net_in_x
    new_col = 'last_'+str(interval)+'_'+time_col+'_'+col
    print 'Parsing %s ...'%new_col
    
    df_data.loc[:, new_col] = 0.0
    for service in set(df_data.service_name):
        cond = (df_data.service_name==service)
        t1 = [] # 存储过去秒数
        t2 = [] # 存储当前秒数
        bi = df_data[cond].index[0]
        for i in df_data[cond].index[1:]:
            if df_data.loc[bi,'sec'] + interval == df_data.loc[i,'sec']:
                # 检测相同服务且相隔10秒的数据行
                t1.append(df_data.loc[bi,'sec'])
                t2.append(df_data.loc[i,'sec'])
            bi = i
        cond1 = cond&(df_data.sec.isin(t1))
        cond2 = cond&(df_data.sec.isin(t2))
        df_data.loc[cond2, new_col] = list(df_data[cond1][col])

def feat_past_time(df_data, col, time_col, interval):
    # **必须先提取last特征**
    # 过去一段时间内的特征值: avg, max, min
    # e.g. past_10_sec_net_in_x_avg
    col = '_' + time_col + '_' + col
    new_col = 'past_' + str(interval) + col
    print 'Parsing %s ...'%new_col
    
    cols = ['last_10'+col]
    for t in range(20, interval+1, 10):
        cols.append('last_'+str(t)+col)
        
    df_data.loc[:,new_col+'_sum'] = list(df_data[cols].sum(axis=1))
    df_data.loc[:,new_col+'_avg'] = list(df_data[cols].mean(axis=1))
    df_data.loc[:,new_col+'_min'] = list(df_data[cols].min(axis=1))
    df_data.loc[:,new_col+'_max'] = list(df_data[cols].max(axis=1))
        
start = time.time()
# 特征工程
# last_10_sec_net_in_x 表示10秒前的net_in值
for interval in range(10, 31, 10):
    # 提取半分钟前每隔10s的特征值
    feat_last_time(new_df, 'mem_use_x', 'sec', interval)
    feat_last_time(new_df, 'mem_use_y', 'sec', interval)
    feat_last_time(new_df, 'net_in_x', 'sec', interval)
    feat_last_time(new_df, 'net_in_y', 'sec', interval)
    feat_last_time(new_df, 'net_out_x', 'sec', interval)
    feat_last_time(new_df, 'net_out_y', 'sec', interval)
    feat_last_time(new_df, 'disk_in', 'sec', interval)
    feat_last_time(new_df, 'disk_out', 'sec', interval)
    feat_last_time(new_df, 'bi', 'sec', interval)
    feat_last_time(new_df, 'bo', 'sec', interval)
    feat_last_time(new_df, 'cpu_userate', 'sec', interval)
    feat_last_time(new_df, 'container_num', 'sec', interval)
    feat_last_time(new_df, 'image_num', 'sec', interval)
    feat_last_time(new_df, 'disk_use', 'sec', interval)
    feat_last_time(new_df, 'min', 'sec', interval)
    feat_last_time(new_df, 'mem_rate', 'sec', interval)
    feat_last_time(new_df, 'disk_rate', 'sec', interval)
    
for interval in range(20, 31, 10):
    # 提取半分钟前每隔10s的特征值
    feat_past_time(new_df, 'mem_use_x', 'sec', interval)
    feat_past_time(new_df, 'mem_use_y', 'sec', interval)
    feat_past_time(new_df, 'net_in_x', 'sec', interval)
    feat_past_time(new_df, 'net_in_y', 'sec', interval)
    feat_past_time(new_df, 'net_out_x', 'sec', interval)
    feat_past_time(new_df, 'net_out_y', 'sec', interval)
    feat_past_time(new_df, 'disk_in', 'sec', interval)
    feat_past_time(new_df, 'disk_out', 'sec', interval)
    feat_past_time(new_df, 'bi', 'sec', interval)
    feat_past_time(new_df, 'bo', 'sec', interval)
    feat_past_time(new_df, 'cpu_userate', 'sec', interval)
    feat_past_time(new_df, 'container_num', 'sec', interval)
    feat_past_time(new_df, 'image_num', 'sec', interval)
    feat_past_time(new_df, 'disk_use', 'sec', interval)
    feat_past_time(new_df, 'min', 'sec', interval)
    feat_past_time(new_df, 'mem_rate', 'sec', interval)
    feat_past_time(new_df, 'disk_rate', 'sec', interval)
print 'Feature Engineering cost: %.6fs'%(time.time() - start)

import xgboost as xgb
from sklearn.model_selection import train_test_split
# 划分数据集：假设所有提取得过去数据为X，预测得y值为 mem_use_x
col_y = 'mem_use_x'
col_X = []
for col in new_df.columns:
    if col.find('past') >= 0 or col.find('last') >= 0:
        # 只使用特征工程得到得过去特征
        col_X.append(col)
X_train, X_test, y_train, y_test = train_test_split(new_df[col_X], new_df[col_y], 
                                                    test_size=0.3, random_state=42)
print np.shape(X_train), np.shape(y_train)

reg = xgb.XGBRegressor(  max_depth=3, 
                         learning_rate=0.1, 
                         n_estimators=100, 
                         silent=True, 
                         objective='reg:linear', 
                         booster='gbtree', 
                         n_jobs=6, 
                         min_child_weight=1, 
                         max_delta_step=0, 
                         subsample=1, 
                         colsample_bytree=1, 
                         colsample_bylevel=1, 
                         reg_alpha=0, 
                         reg_lambda=1, 
                         scale_pos_weight=1, 
                         base_score=0.5, 
                         random_state=0, 
                         seed=10)
# 训练模型
reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse') # 使用均方误差rmse作为评价指标
# 预测结果
print '测试数据预测结果（前5）:'
print reg.predict(X_test)[:5]
print '测试数据真实结果（前5）:'
print y_test[:5]

'''
(56081, 187) (56081,)
[0]	validation_0-rmse:534459
[1]	validation_0-rmse:481320
/usr/local/lib/python2.7/dist-packages/xgboost-0.6-py2.7.egg/xgboost/sklearn.py:203: DeprecationWarning: The seed parameter is deprecated as of version .6.Please use random_state instead.seed is deprecated.
  'seed is deprecated.', DeprecationWarning)
[2]	validation_0-rmse:433546
[3]	validation_0-rmse:390490
[4]	validation_0-rmse:351781
[5]	validation_0-rmse:316964
[6]	validation_0-rmse:285667
[7]	validation_0-rmse:257431
[8]	validation_0-rmse:232090
[9]	validation_0-rmse:209240
[10]	validation_0-rmse:188687
[11]	validation_0-rmse:170231
[12]	validation_0-rmse:153639
[13]	validation_0-rmse:138729
[14]	validation_0-rmse:125351
[15]	validation_0-rmse:113334
[16]	validation_0-rmse:102588
[17]	validation_0-rmse:92949.6
[18]	validation_0-rmse:84302.7
[19]	validation_0-rmse:76590.6
[20]	validation_0-rmse:69722.1
[21]	validation_0-rmse:63571.2
[22]	validation_0-rmse:58142.2
[23]	validation_0-rmse:53334.4
[24]	validation_0-rmse:49085.7
[25]	validation_0-rmse:45311
[26]	validation_0-rmse:42036.4
[27]	validation_0-rmse:39198.6
[28]	validation_0-rmse:36705.2
[29]	validation_0-rmse:34567
[30]	validation_0-rmse:32733.6
[31]	validation_0-rmse:31167.6
[32]	validation_0-rmse:29839.4
[33]	validation_0-rmse:28702.5
[34]	validation_0-rmse:27766.8
[35]	validation_0-rmse:26981.8
[36]	validation_0-rmse:26324.7
[37]	validation_0-rmse:25773.2
[38]	validation_0-rmse:25314.6
[39]	validation_0-rmse:24932.1
[40]	validation_0-rmse:24626.7
[41]	validation_0-rmse:24373.1
[42]	validation_0-rmse:24131.9
[43]	validation_0-rmse:23951.5
[44]	validation_0-rmse:23807.4
[45]	validation_0-rmse:23684.6
[46]	validation_0-rmse:23583.4
[47]	validation_0-rmse:23502.9
[48]	validation_0-rmse:23440.9
[49]	validation_0-rmse:23357.9
[50]	validation_0-rmse:23320.1
[51]	validation_0-rmse:23292
[52]	validation_0-rmse:23266.4
[53]	validation_0-rmse:23242.4
[54]	validation_0-rmse:23201.4
[55]	validation_0-rmse:23177.5
[56]	validation_0-rmse:23164.2
[57]	validation_0-rmse:23114.3
[58]	validation_0-rmse:23108
[59]	validation_0-rmse:23066.2
[60]	validation_0-rmse:23059.9
[61]	validation_0-rmse:23055.5
[62]	validation_0-rmse:23016.6
[63]	validation_0-rmse:23015.3
[64]	validation_0-rmse:23020.6
[65]	validation_0-rmse:23020.1
[66]	validation_0-rmse:23020.1
[67]	validation_0-rmse:23012.8
[68]	validation_0-rmse:23011.9
[69]	validation_0-rmse:23002.9
[70]	validation_0-rmse:23002.2
[71]	validation_0-rmse:22980.2
[72]	validation_0-rmse:22988.1
[73]	validation_0-rmse:22990
[74]	validation_0-rmse:22984.3
[75]	validation_0-rmse:22993.3
[76]	validation_0-rmse:22974.5
[77]	validation_0-rmse:22956.4
[78]	validation_0-rmse:22960.3
[79]	validation_0-rmse:22962.8
[80]	validation_0-rmse:22937.3
[81]	validation_0-rmse:22927.4
[82]	validation_0-rmse:22923.8
[83]	validation_0-rmse:22910.1
[84]	validation_0-rmse:22912.4
[85]	validation_0-rmse:22913.3
[86]	validation_0-rmse:22903.2
[87]	validation_0-rmse:22902.6
[88]	validation_0-rmse:22897.9
[89]	validation_0-rmse:22872.4
[90]	validation_0-rmse:22873.2
[91]	validation_0-rmse:22876.9
[92]	validation_0-rmse:22876.5
[93]	validation_0-rmse:22891
[94]	validation_0-rmse:22890.7
[95]	validation_0-rmse:22890.4
[96]	validation_0-rmse:22883.6
[97]	validation_0-rmse:22881.1
[98]	validation_0-rmse:22896.8
[99]	validation_0-rmse:22895.9
测试数据预测结果（前5）:
[	91621.640625    
	516542.8125       
	38499.62109375  
	612720.9375
    7740.74707031]
测试数据真实结果（前5）:
77148     89057.281250
79740    515891.187500
40333     38236.160156
66405    613376.000000
24864      6563.839844
Name: mem_use_x, dtype: float32
'''

