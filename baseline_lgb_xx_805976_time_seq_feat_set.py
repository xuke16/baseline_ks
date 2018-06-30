#coding:utf-8

# 简单分析：
# 这份代码和特征主要考虑了用户最近一段时间的行为，
# 设计了一个权值银子去计算用户在最近一段时间的访问特征
# 因此，可能时间窗口的特征过于强，反而对于很多统计特征的表达并不好
# 当前这份代码，线下线上成绩约等于 0.80625 - 0.8206 附近
# 不可考究是否过拟合或者其他，需要等B榜验证成绩

import pandas as pd
import numpy as np
np.random.seed(42)
import pickle
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
# 数据采样粒度为day
import os
import time
start_time = time.time()

# 读取原始数据
def read_data():
    reshape_data_user_set = []
    if os.path.exists('../cache/user_info_register_and_day.pkl'):
        print('read user_info_register_and_day from pkl')
        user_info_register_and_day = pickle.load(open('../cache/user_info_register_and_day.pkl','rb'))
    else:
        user_register_log = pd.read_csv('../data/user_register_log.txt', sep='\t', header=None,
                                        dtype={0: np.str, 1: np.int, 2: np.int, 3: np.int})
        user_register_log.columns = ['user_id', 'register_day', 'register_type', 'device_type']

        reshape_data_1 = user_register_log[user_register_log['register_day']<22]
        reshape_data_2 = user_register_log[user_register_log['register_day']>24]
        # 随机采样
        reshape_data_3 = user_register_log[user_register_log['register_day']==22].sample(frac=1)
        reshape_data_4 = user_register_log[user_register_log['register_day']==23].sample(frac=1)
        reshape_data_5 = user_register_log[user_register_log['register_day']==24].sample(frac=1)

        reshape_data = pd.concat([reshape_data_1,reshape_data_2])
        reshape_data = pd.concat([reshape_data,reshape_data_3])
        reshape_data = pd.concat([reshape_data,reshape_data_4])
        reshape_data = pd.concat([reshape_data,reshape_data_5])

        reshape_data = reshape_data.reset_index(drop=True)

        reshape_data_user_set = set(reshape_data['user_id'].unique())
        # 根据用户注册信息补全到30号包括30号的用户行为信息
        user_info_register_and_day = pd.DataFrame()
        days = []
        register_days = []
        user_ids = []
        register_types = []
        device_types = []

        for i in reshape_data.values:
            day = int(i[1])
            register_day = int(i[1])
            user_id = i[0]
            register_type = i[2]
            device_type = i[3]
            for j in range(day, 30 + 1):
                days.append(j)
                register_days.append(register_day)
                user_ids.append(user_id)
                register_types.append(register_type)
                device_types.append(device_type)

        del user_register_log

        user_info_register_and_day['user_id'] = user_ids
        user_info_register_and_day['register_day'] = register_days
        user_info_register_and_day['register_type'] = register_types
        user_info_register_and_day['device_type'] = device_types
        user_info_register_and_day['day'] = days

        pickle.dump(user_info_register_and_day,open('../cache/user_info_register_and_day.pkl','wb'))

    if os.path.exists('../cache/app_launch_log.pkl'):
        print('read app_launch_log from pkl')
        app_launch_log = pickle.load(open('../cache/app_launch_log.pkl','rb'))
    else:
        app_launch_log = pd.read_csv('../data/app_launch_log.txt', sep='\t', header=None,
                                     dtype={0: np.str, 1: np.int})
        app_launch_log.columns = ['user_id', 'day']

        app_launch_log = app_launch_log[app_launch_log['user_id'].isin(reshape_data_user_set)].reset_index(drop=True)

        app_launch_log['app_launch_log_flag'] = 1
        pickle.dump(app_launch_log, open('../cache/app_launch_log.pkl', 'wb'))

    if os.path.exists('../cache/video_create_log.pkl'):
        print('read video_create_log from pkl')
        video_create_log = pickle.load(open('../cache/video_create_log.pkl','rb'))
    else:
        video_create_log = pd.read_csv('../data/video_create_log.txt', sep='\t', header=None,
                                       dtype={0: np.str, 1: np.int})
        video_create_log.columns = ['user_id', 'day']

        video_create_log = video_create_log[video_create_log['user_id'].isin(reshape_data_user_set)].reset_index(drop=True)

        video_create_log['video_flag'] = 1
        pickle.dump(video_create_log, open('../cache/video_create_log.pkl', 'wb'))

    if os.path.exists('../cache/user_activity_log.pkl'):
        print('read user_activity_log from pkl')
        user_activity_log = pickle.load(open('../cache/user_activity_log.pkl','rb'))
    else:
        user_activity_log = pd.read_csv('../data/user_activity_log.txt', sep='\t', header=None,
                                        dtype={0: np.str, 1: np.int, 2: np.int, 3: np.int, 4: np.int, 5: np.int})
        user_activity_log.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']

        user_activity_log = user_activity_log[user_activity_log['user_id'].isin(reshape_data_user_set)].reset_index(drop=True)

        user_activity_log['user_activity_log_flag'] = 1
        pickle.dump(user_activity_log, open('../cache/user_activity_log.pkl', 'wb'))

    return user_info_register_and_day,app_launch_log,video_create_log,user_activity_log
# 获取数据的标签轴
def get_label(data_set,end_time,time_wide):
    '''
    :param data_set: user_info_register_and_day,app_launch_log,video_create_log,user_activity_log 活跃日志提取用户
    :param end_time: 结束的日期
    :param time_wide: 时间往前偏移量
    :return: 用户集合 begin_time
    '''
    date_range = [x+1 for x in range(end_time-time_wide,end_time)]
    register_date = date_range[0]

    # 创建四个用户集合
    register_user_set = set()

    # 1.筛选 end_tiem - time_wide 之前注册的用户
    register_user = data_set[0][data_set[0]['register_day'] < register_date]
    if sorted(register_user['register_day'].unique())[-1] < register_date:
        register_user_set = set(register_user['user_id'].unique())
        print(sorted(register_user['register_day'].unique())[-1], 'day before register_user user number', len(register_user_set))

    # 2.1 获取活跃用户的集合
    app_launch_user = data_set[1][data_set[1]['day'].isin(date_range)]
    # if (sorted(app_launch_user['day'].unique())== date_range):
    app_launch_user_set = set(app_launch_user['user_id'].unique())
    print(sorted(app_launch_user['day'].unique())[-1], sorted(app_launch_user['day'].unique())[0],
              'day before app_launch_user user number', len(app_launch_user_set))

    # 2.2 获取活跃用户的集合
    video_create_log = data_set[2][data_set[2]['day'].isin(date_range)]
    # if (video_create_log['day'].unique()== date_range):
    video_create_user_set = set(video_create_log['user_id'].unique())
    print(sorted(video_create_log['day'].unique())[-1], sorted(video_create_log['day'].unique())[0],
          'day before video_create_log user number', len(video_create_user_set))

    # 2.3 获取活跃用户的集合
    user_activity_log = data_set[3][data_set[3]['day'].isin(date_range)]
    # if (user_activity_log['day'].unique()== date_range):
    user_activity_user_set = set(user_activity_log['user_id'].unique())
    print(sorted(user_activity_log['day'].unique())[-1], sorted(user_activity_log['day'].unique())[0],
          'day before user_activity_log user number', len(user_activity_user_set))

    user_set = register_user_set & (app_launch_user_set | video_create_user_set | user_activity_user_set)
    print('the future 7 day activate user',len(user_set))
    return user_set,register_date - 1
# 线下成绩计算公式
def get_score(pre_user,true_user,day=0.0):
    pre_user_set = set(pre_user)
    true_user_set = set(true_user)
    print(len(pre_user_set))
    print(len(true_user_set))
    try:
        precision = len(pre_user_set&true_user_set)* 1.0 / len(pre_user_set)
        recall = len(pre_user_set&true_user_set)* 1.0 / len(true_user_set)
        print('%f precision %f'%(day,precision))
        print('%f recall %f'%(day,recall))
        F1_Score = (2.0 * precision * recall) / (precision + recall)
        print('%f F1 %f'%(day,F1_Score))
        return F1_Score
    except:
        print('0 user')
        return 0
# 计算样本数据在真实值的概率
def get_xx_cover_radio(label,train_data):
    print(u'all data of user',len(set(train_data)))
    print(u'day user of data',len(set(label)))
    train_data_in_label = len(set(label)&set(train_data)) * 1.0 / len(set(label))
    get_score(set(train_data),set(label))
    print(train_data_in_label)
# 获取样本
def get_sample(data_set,sample_time,true_user_set,time_wide=0):
    date_range = [x for x in range(sample_time-time_wide,sample_time+1)]
    print(date_range)
    sample_user_basic_info = data_set[0][data_set[0]['day'].isin(date_range)]
    sample_user_basic_info.loc[sample_user_basic_info['user_id'].isin(list(true_user_set)),'target'] = 1
    sample_user_basic_info['target'] = sample_user_basic_info['target'].fillna(0)
    # 计算覆盖和全集F1
    get_xx_cover_radio(true_user_set,sample_user_basic_info['user_id'])
    return sample_user_basic_info

from itertools import groupby
def get_max_seq_l(l):
    l = sorted(l)
    fun = lambda x: x[1] - x[0]
    max_seq_tmp = []

    for k, g in groupby(enumerate(l), fun):
        max_seq_tmp.append([v for i, v in g])

    return ([x for x in range(max(max_seq_tmp)[0],30+1)][-1] - max(max_seq_tmp)[-1])


from scipy.stats import mode
def get_mode(data):
    # print(mode(data)[0][0])
    return mode(data)[0][0]

# 获取特征
def make_feat(data_set,sample):
    '''
    :param data_set:
    :param sample:
    :return:
    data_set = [user_info_register_and_day, app_launch_log, video_create_log, user_activity_log]
    '''

    # 简单的数据采样分析
    #       register_day  register_type  device_type     0
    # 18147            24              3            1  1107
    # 18178            24              4            1    21
    # -----------------------------------------------------
    #        register_day  register_type  device_type  0
    # 18908            25              3            1  8
    # -----------------------------------------------------
    #        register_day  register_type  device_type  0
    # 19704            26              3            1  4

    # 83 223

    feat_sample = sample.copy()

    sample_date = sample['day'].unique()[0]
    print('sample_data', sample['day'].unique())

    sample_copy = pd.DataFrame()

    days = []
    register_days = []
    user_ids = []
    register_types = []
    device_types = []
    # 重构 sample data
    for i in sample.values:
        register_day = int(i[1])
        user_id = i[0]
        register_type = i[2]
        device_type = i[3]

        for j in range(max(sample_date - 14,register_day), sample_date + 1):
            days.append(j)
            register_days.append(register_day)
            user_ids.append(user_id)
            register_types.append(register_type)
            device_types.append(device_type)

    sample_copy['user_id'] = user_ids
    sample_copy['register_day'] = register_days
    sample_copy['register_type'] = register_types
    sample_copy['device_type'] = device_types
    sample_copy['day'] = days
    sample_copy['user_id'] = sample_copy['user_id'].astype(str)
    print('-------------------------------------------')
    print('reshape_sample_data_shape',sample_copy.shape)
    print('-------------------------------------------')
    sample['user_id'] = sample['user_id'].astype(str)
    data_set[1]['user_id'] = data_set[1]['user_id'].astype(str)
    data_set[2]['user_id'] = data_set[2]['user_id'].astype(str)
    data_set[3]['user_id'] = data_set[3]['user_id'].astype(str)
    data_set[0]['user_id'] = data_set[0]['user_id'].astype(str)

    sample = pd.merge(sample_copy,sample,on=['user_id','register_day','register_type','device_type','day'],how='left',copy=False)

    # 采取半个月的用户历史特征
    time_scale = [x for x in range(sample_date - 14, sample_date + 1)]
    print('特征采集范围',sorted(time_scale),len(time_scale))

    launch_feat_ = data_set[1][data_set[1]['day'].isin(time_scale)]
    sample = pd.merge(sample, launch_feat_, on=['user_id','day'], how='left', copy=False)
    sample['app_launch_log_flag'] = sample['app_launch_log_flag'].fillna(0)

    # bug 但是提分了，用户每天可能多次启动APP进行拍摄，直接与启动日志拼接，比去重后分数有所提高
    # 考虑到发生这样的bug，认为每次拍摄行为，应该是出发一次启动APP的日志
    video_create_feat_ = data_set[2][data_set[2]['day'].isin(time_scale)]
    video_create_feat_copy = data_set[2][data_set[2]['day'].isin(time_scale)]

    # video_create_feat_ = video_create_feat_.groupby(['user_id', 'day'])['video_flag'].sum().reset_index()
    # video_create_feat_.columns = ['user_id', 'day', 'video_flag']

    sample = pd.merge(sample, video_create_feat_, on=['user_id', 'day'], how='left', copy=False)
    sample['video_flag'] = sample['video_flag'].fillna(0)

    user_activity_feat_ = data_set[3][data_set[3]['day'].isin(time_scale)][['user_id','day','user_activity_log_flag']].drop_duplicates()
    user_activity_feat_copy = data_set[3][data_set[3]['day'].isin(time_scale)]

    # 提取用户序列行为的特征集合
    # print(user_activity_feat_copy.columns)
    user_activity_feat_f = user_activity_feat_copy.copy()
    user_activity_feat_f['video_id'] = user_activity_feat_f['video_id'].astype(str)
    user_video_seq_f = user_activity_feat_f.groupby(['user_id'])['video_id'].apply(lambda x:' '.join(list(set(x)))).reset_index()
    user_video_seq_f.columns = ['user_id','video_seq_']

    sample = pd.merge(sample, user_activity_feat_, on=['user_id', 'day'], how='left', copy=False)
    sample['user_activity_log_flag'] = sample['user_activity_log_flag'].fillna(0)

    # 以下为增加权重系数的特征组合
    # 增加权重系数
    sample['weight'] = sample_date + 1 - sample['day']
    sample['weight'] = 1 / sample['weight']

    sample['app_launch_log_flag'] = sample['app_launch_log_flag'] * sample['weight']
    sample['video_flag'] = sample['video_flag'] * sample['weight']
    sample['user_activity_log_flag'] = sample['user_activity_log_flag'] * sample['weight']
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print('before sample shape',sample.shape)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    ###############################################################################

    # 最近 7 天的用户行为 和 权重
    sample_copy['day'] = sample_copy['day'].astype(int)
    sample_copy_seven = sample_copy[sample_copy['day'] >= (sample_date - 7)]
    print(sample_copy_seven.shape)

    time_scale_seven = [x for x in range(sample_date - 7, sample_date + 1)]

    launch_feat_seven = data_set[1][data_set[1]['day'].isin(time_scale_seven)]
    launch_feat_seven = launch_feat_seven.sort_values(['user_id','day'])

    sample_seven_s = pd.merge(sample_copy_seven, launch_feat_seven, on=['user_id', 'day'], how='left', copy=False)
    sample_seven_s['app_launch_log_flag'] = sample_seven_s['app_launch_log_flag'].fillna(0)

    video_create_feat_7 = data_set[2][data_set[2]['day'].isin(time_scale_seven)]

    # video_create_feat_7 = video_create_feat_7.groupby(['user_id', 'day'])['video_flag'].sum().reset_index()
    # video_create_feat_7.columns = ['user_id', 'day', 'video_flag']

    sample_seven_s = pd.merge(sample_seven_s, video_create_feat_7, on=['user_id', 'day'], how='left', copy=False)
    sample_seven_s['video_flag'] = sample_seven_s['video_flag'].fillna(0)

    user_activity_feat_7 = data_set[3][data_set[3]['day'].isin(time_scale_seven)][['user_id', 'day', 'user_activity_log_flag']].drop_duplicates()

    sample_seven_s = pd.merge(sample_seven_s, user_activity_feat_7, on=['user_id', 'day'], how='left', copy=False)
    sample_seven_s['user_activity_log_flag'] = sample_seven_s['user_activity_log_flag'].fillna(0)

    sample_seven = sample_seven_s[['user_id','day','app_launch_log_flag','video_flag','user_activity_log_flag']]


    sample_seven['weight'] = sample_date + 1 - sample_seven['day']
    sample_seven['weight'] = 1 / sample_seven['weight']

    sample_seven['app_launch_log_flag'] = sample_seven['app_launch_log_flag'] * sample_seven['weight']
    sample_seven['video_flag'] = sample_seven['video_flag'] * sample_seven['weight']
    sample_seven['user_activity_log_flag'] = sample_seven['user_activity_log_flag'] * sample_seven['weight']

    print(sample_seven.shape)

    ###############################################################################
    # 3 days
    sample_copy_3 = sample_copy[sample_copy['day'] >= (sample_date - 3)]

    print(sample_copy_3.shape)

    time_scale_3 = [x for x in range(sample_date - 3, sample_date + 1)]

    launch_feat_3 = data_set[1][data_set[1]['day'].isin(time_scale_3)]
    launch_feat_3 = launch_feat_3.sort_values(['user_id','day'])

    sample_seven_s3 = pd.merge(sample_copy_3, launch_feat_3, on=['user_id', 'day'], how='left', copy=False)
    sample_seven_s3['app_launch_log_flag'] = sample_seven_s3['app_launch_log_flag'].fillna(0)

    video_create_feat_3 = data_set[2][data_set[2]['day'].isin(time_scale_3)]

    # video_create_feat_3 = video_create_feat_3.groupby(['user_id', 'day'])['video_flag'].sum().reset_index()
    # video_create_feat_3.columns = ['user_id', 'day', 'video_flag']

    sample_seven_s3 = pd.merge(sample_seven_s3, video_create_feat_3, on=['user_id', 'day'], how='left', copy=False)
    sample_seven_s3['video_flag'] = sample_seven_s3['video_flag'].fillna(0)

    user_activity_feat_3 = data_set[3][data_set[3]['day'].isin(time_scale_3)][
        ['user_id', 'day', 'user_activity_log_flag']].drop_duplicates()
    sample_seven_s3 = pd.merge(sample_seven_s3, user_activity_feat_3, on=['user_id', 'day'], how='left', copy=False)
    sample_seven_s3['user_activity_log_flag'] = sample_seven_s3['user_activity_log_flag'].fillna(0)

    sample_seven_s3 = sample_seven_s3[['user_id','day','app_launch_log_flag','video_flag','user_activity_log_flag']]

    sample_seven_s3['weight'] = sample_date + 1 - sample_seven_s3['day']
    sample_seven_s3['weight'] = 1 / sample_seven_s3['weight']

    sample_seven_s3['app_launch_log_flag'] = sample_seven_s3['app_launch_log_flag'] * sample_seven_s3['weight']
    sample_seven_s3['video_flag'] = sample_seven_s3['video_flag'] * sample_seven_s3['weight']
    sample_seven_s3['user_activity_log_flag'] = sample_seven_s3['user_activity_log_flag'] * sample_seven_s3['weight']

    print(sample_seven_s3.shape)
    ###############################################################################

    # 1.增加时间权重特征
    # 1.1.窗口内用户 user_activity_log_flag 的频次
    user_activate_c_feat_in_windows = sample.groupby(['user_id'])['user_activity_log_flag'].sum().reset_index()
    user_activate_c_feat_in_windows.columns = ['user_id', 'windows_user_activity_feat']

    # 1.2.窗口内用户video_flag的频次
    user_video_feat_in_windows = sample.groupby(['user_id'])['video_flag'].sum().reset_index()
    user_video_feat_in_windows.columns = ['user_id', 'windows_create_video_feat']

    # 1.3.用户在一个窗口期内的访问日访问频次特征
    user_activate_feat_in_windows = sample.groupby(['user_id'])['app_launch_log_flag'].sum().reset_index()
    user_activate_feat_in_windows.columns = ['user_id','windows_launch_feat']

    # 1.4 windows 7 特征统计
    user_activate_7_feat_in_windows = sample_seven.groupby(['user_id'])['app_launch_log_flag'].sum().reset_index()
    user_activate_7_feat_in_windows.columns = ['user_id', 'windows_7_launch_feat']

    # 1.6 windows 3
    user_activate_3_feat_in_windows = sample_seven_s3.groupby(['user_id'])['app_launch_log_flag'].sum().reset_index()
    user_activate_3_feat_in_windows.columns = ['user_id', 'windows_3_launch_feat']

    # user_log_3_feat_in_windows = sample_seven_s3.groupby(['user_id'])['user_activity_log_flag'].sum().reset_index()
    # user_log_3_feat_in_windows.columns = ['user_id', 'user_activity_log_flag']

    # feat_sample = pd.merge(feat_sample,user_log_3_feat_in_windows,on='user_id',how='left',copy=False)

    feat_sample = pd.merge(feat_sample,user_activate_feat_in_windows,on='user_id',how='left',copy=False)
    del user_activate_feat_in_windows

    feat_sample = pd.merge(feat_sample,user_video_feat_in_windows,on='user_id',how='left',copy=False)
    del user_video_feat_in_windows

    feat_sample = pd.merge(feat_sample,user_activate_c_feat_in_windows,on='user_id',how='left',copy=False)
    del user_activate_c_feat_in_windows
    # 最近7天行为
    feat_sample = pd.merge(feat_sample,user_activate_7_feat_in_windows,on='user_id',how='left',copy=False)
    del user_activate_7_feat_in_windows
    # 最近3天行为
    feat_sample = pd.merge(feat_sample,user_activate_3_feat_in_windows,on='user_id',how='left',copy=False)
    del user_activate_3_feat_in_windows

    # 1.4. 二次特征组合
    feat_sample['bayes_f1'] = feat_sample['windows_create_video_feat'] / (feat_sample['windows_launch_feat'] + 0.00001)
    feat_sample['bayes_f2'] = feat_sample['windows_user_activity_feat'] / (feat_sample['windows_launch_feat'] + 0.00001)

    feat_sample['bayes_f3'] = feat_sample['windows_7_launch_feat'] / (feat_sample['windows_launch_feat'] + 0.00001)
    feat_sample['bayes_f4'] = (feat_sample['windows_launch_feat'] - feat_sample['windows_7_launch_feat']) / (feat_sample['windows_launch_feat'] + 0.00001)

    # add feat 用户最近一次启动的时间的时间差
    user_current_launch_app_time_gap = launch_feat_.groupby(['user_id']).day.max().reset_index()
    user_current_launch_app_time_gap['current_onec_launch'] = sample_date - user_current_launch_app_time_gap['day'] + 1
    feat_sample = pd.merge(feat_sample, user_current_launch_app_time_gap[['user_id', 'current_onec_launch']],on=['user_id'], how='left', copy=False)
    feat_sample['current_onec_launch'] = feat_sample['current_onec_launch'].fillna(17 + 1)

    #
    # user_current_launch_app_time_gap = launch_feat_.groupby(['user_id']).day.min().reset_index()
    # user_current_launch_app_time_gap['last_onec_launch'] =user_current_launch_app_time_gap['day']
    # feat_sample = pd.merge(feat_sample, user_current_launch_app_time_gap[['user_id', 'last_onec_launch']],
    #                        on=['user_id'], how='left', copy=False)
    # feat_sample['last_onec_launch'] = feat_sample['last_onec_launch'].fillna(-1)

    # action page 特征
    user_action_f_ = pd.concat([user_activity_feat_copy,pd.get_dummies(user_activity_feat_copy['page'],prefix='page')],axis=1)
    user_action_f_ = pd.concat([user_action_f_,pd.get_dummies(user_activity_feat_copy['action_type'],prefix='action_type')],axis=1)

    # 视频不同行为的特征
    # 视频的统计特征  用户对视频统计特征的统计 action
    video_windwos_feat_ = user_action_f_.groupby(['video_id'])[['action_type_0', 'action_type_1', 'action_type_2',
                                                                'action_type_3', 'action_type_4', 'action_type_5']].mean().reset_index()

    video_windwos_feat_ = pd.merge(user_action_f_[['user_id','video_id']].copy(),video_windwos_feat_,on=['video_id'],how='left',copy=False)

    video_windwos_feat_ = video_windwos_feat_.groupby(['user_id'])[['action_type_0', 'action_type_1', 'action_type_2',
                                                                'action_type_3', 'action_type_4','action_type_5']].mean().reset_index()

    feat_sample = pd.merge(feat_sample,video_windwos_feat_,on=['user_id'],how='left',copy=False)
    del video_windwos_feat_

    # 用户习惯访问页页面的特征
    video_page_windwos_feat_ = user_action_f_.groupby(['user_id'])[['page_0', 'page_1', 'page_2',
                                                                'page_3', 'page_4'
                                                                ]].mean().reset_index()

    feat_sample = pd.merge(feat_sample, video_page_windwos_feat_, on=['user_id'], how='left',copy=False)
    del video_page_windwos_feat_

    # 用户启动app的时间差特征
    # launch_feat_1 = launch_feat_.copy()
    #
    # launch_feat_ = launch_feat_.sort_values(['user_id','day'])
    # launch_feat_['time_launch_diff'] = launch_feat_.groupby(['user_id']).day.diff(-1).apply(np.abs)
    # launch_feat_ = launch_feat_[['user_id','time_launch_diff']].dropna()
    # launch_feat_ = launch_feat_.groupby(['user_id'])['time_launch_diff'].agg({
    #     'time_launch_diff_max':np.max,
    #     'time_launch_diff_std':np.std,
    #     'time_launch_diff_median':np.median,
    # }).reset_index()
    #
    # feat_sample = pd.merge(feat_sample, launch_feat_, on=['user_id'], how='left')
    # del launch_feat_
    #
    # # 上2次的访问时间差
    # launch_feat_1 = launch_feat_1.sort_values(['user_id', 'day'])
    # launch_feat_1['time_launch_diff'] = launch_feat_1.groupby(['user_id']).day.diff(-2).apply(np.abs)
    # launch_feat_1 = launch_feat_1[['user_id', 'time_launch_diff']].dropna()
    # launch_feat_1 = launch_feat_1.groupby(['user_id'])['time_launch_diff'].agg({
    #     'time_launch_diff_max1': np.max,
    #     'time_launch_diff_std1': np.std,
    #     'time_launch_diff_median1': np.median
    # }).reset_index()
    #
    # feat_sample = pd.merge(feat_sample, launch_feat_1, on=['user_id'], how='left')

    # 用户 video nunique 特征
    user_activity_f_video = user_activity_feat_copy.groupby(['user_id'])['video_id'].nunique().reset_index()
    user_activity_f_video.columns = ['user_id','nunique_video']

    feat_sample = pd.merge(feat_sample, user_activity_f_video, on=['user_id'], how='left')

    # 用户 author nunique 特征
    user_activity_f_author = user_activity_feat_copy.groupby(['user_id'])['author_id'].nunique().reset_index()
    user_activity_f_author.columns = ['user_id', 'nunique_author']

    feat_sample = pd.merge(feat_sample, user_activity_f_author, on=['user_id'], how='left',copy=False)

    # 用户重复观看作品的次数特征
    # xtmp = user_activity_feat_copy.groupby(['user_id','author_id']).size().reset_index()
    # xtmp.columns = ['user_id','author_id','rep_author']
    # xtmp = xtmp.sort_values(['rep_author','user_id'])
    #
    # xtmp = xtmp.groupby(['user_id'])['rep_author'].agg({
    #     'rep_author_mean':np.mean,
    #     'rep_author_std':np.std,
    # })
    #
    # feat_sample = pd.merge(feat_sample, xtmp, on=['user_id'], how='left',copy=False)

    # windows内的视频每天被多少人观看 除去当前用户本身
    # authored_user = user_activity_feat_copy.groupby(['video_id'])['user_id'].nunique().reset_index()
    # authored_user.columns = ['video_id','user_nunique']
    # authored_user = pd.merge(user_activity_feat_copy[['user_id','video_id']],authored_user,on=['video_id'],how='left',copy=False)
    # authored_user['user_nunique'] = authored_user['user_nunique'] - 1
    # authored_user = authored_user.groupby(['user_id'])['user_nunique'].agg({
    #     'user_nunique_mean':np.mean,
    #     'user_nunique_std':np.std,
    # }).reset_index()
    #
    # feat_sample = pd.merge(feat_sample,authored_user,on=['user_id'],how='left',copy=False)
    # del user_action_f_
    #
    # feat_sample = pd.merge(feat_sample,user_video_seq_f,on=['user_id'],how='left',copy=False)

    # 注册类型的平均访问时间特征
    # register_launch_feat = feat_sample.groupby(['register_type']).windows_launch_feat.agg({
    #     'register_type_mean':np.mean,
    #     # 'register_type_median':np.median
    # }).reset_index()
    # # register_launch_feat.columns = ['register_type','register_launch_feat']
    #
    # feat_sample = pd.merge(feat_sample,register_launch_feat,on=['register_type'],how='left',copy=False)

    # register_launch_feat = feat_sample.groupby(['device_type']).windows_launch_feat.mean().reset_index()
    # register_launch_feat.columns = ['device_type', 'register_launch_feat']
    #
    # feat_sample = pd.merge(feat_sample, register_launch_feat, on=['device_type'], how='left', copy=False)
    # #
    # user_create_video_times = video_create_feat_copy.groupby(['user_id']).size().reset_index()
    # user_create_video_times.columns = ['user_id','user_create_video_times']
    # feat_sample = pd.merge(feat_sample,user_create_video_times,on=['user_id'],how='left',copy=False)
    #

    # user_current_launch_app_time_gap = sample[sample['app_launch_log_flag'] >= 1].groupby(['user_id']).day.max().reset_index()
    # user_current_launch_app_time_gap['current_onec_launch'] = sample_date - user_current_launch_app_time_gap['day'] + 1
    # feat_sample = pd.merge(feat_sample, user_current_launch_app_time_gap[['user_id', 'current_onec_launch']],on=['user_id'], how='left', copy=False)
    # feat_sample['current_onec_launch'] = feat_sample['current_onec_launch'].fillna(sample_date+1)
    #
    # 用户观看的视频频次
    # user_video_size_ = user_activity_feat_copy.groupby(['user_id'])['video_id'].size().reset_index()
    # user_video_size_.columns = ['user_id', 'user_video_size_']
    # feat_sample = pd.merge(feat_sample, user_video_size_, on=['user_id'], how='left', copy=False)

    # 用户每天观看视频的频次
    # user_video_size = user_activity_feat_copy.groupby(['user_id', 'day'])['video_id'].size().reset_index()
    # user_video_size.columns = ['user_id', 'day', 'user_video_size']
    # user_video_size = user_video_size.groupby(['user_id']).user_video_size.agg({
    #     'user_video_size_mean': np.mean,
    #     # 'user_video_size_std': np.std,
    # }).reset_index()
    # feat_sample = pd.merge(feat_sample, user_video_size, on=['user_id'], how='left', copy=False)

    # 用户每天视频的重复次数
    # user_video_day_size = user_activity_feat_copy.groupby(['user_id', 'day', 'video_id']).size().reset_index()
    # user_video_day_size.columns = ['user_id', 'day', 'video_id', 'user_video_day_size']
    # user_video_day_size = user_video_day_size.groupby(['user_id']).user_video_day_size.agg({
    #     'user_video_day_size_mean': np.mean,
    #     'user_video_day_size_std': np.std,
    # }).reset_index()
    # feat_sample = pd.merge(feat_sample, user_video_day_size, on=['user_id'], how='left', copy=False)

    # for day in [1, 2, 3, 5, 7, 14, 21]:
    #     tmp = user_activity_feat_copy[user_activity_feat_copy['day'] > sample_date - day]
    #     print(sorted(tmp['day'].unique()))
    #     day_tmp = tmp.groupby(['user_id'])['video_id'].size().reset_index()
    #     day_tmp.columns = ['user_id', 'video_id_before_%d' % (day)]
    #     feat_sample = pd.merge(feat_sample, day_tmp, on=['user_id'], how='left', copy=False)
    #     feat_sample['video_id_before_%d' % (day)] = feat_sample['video_id_before_%d' % (day)].fillna(0)

    print(len(feat_sample['user_id'].unique()),feat_sample.shape)

    return feat_sample

def make_category_feat(train_val_test):
    return train_val_test


def get_train_val_test():
    user_info_register_and_day, app_launch_log, video_create_log, user_activity_log = read_data()
    data_set = [user_info_register_and_day, app_launch_log, video_create_log, user_activity_log]

    # 1.提交数据的样本集合
    print('===================sub sample===================')
    sub_sample = get_sample(data_set, 30, set(data_set[0][data_set[0]['day']==30]['user_id']))

    # 特征
    sub_sample = make_feat(data_set,sub_sample)

    # 2.线下验证/训练数据集
    train_val = pd.DataFrame()
    # 构造三组样本
    # 样本区间
    # 特征区间 9 样本标签 7
    # 样本 0 22-30 | 31-37
    # 样本 1 15-23 | 24-30
    # 样本 2 08-16 | 17-23
    # 样本 3 01-09 | 10-16
    # 样本 1 2 3
    label_dict = {}
    lllll = [7,7]
    for c, i in enumerate([0, 7]):
        print('===================windows %d==================='%(i))
        true_user_set, sample_day = get_label(data_set, 30-i, lllll[c])
        sample_and_basic_user_info = get_sample(data_set,sample_day,true_user_set)
        # 特征
        sample_and_basic_user_info = make_feat(data_set, sample_and_basic_user_info)

        label_dict[30 - i - lllll[c]] = true_user_set
        train_val = pd.concat([train_val,sample_and_basic_user_info],axis=0)
    #
    print('train_val',train_val['day'].unique())
    print('sub_sample',sub_sample['day'].unique())
    #
    train_val_test = pd.concat([train_val,sub_sample],axis=0)

    train_val_test = make_category_feat(train_val_test)

    return train_val_test,label_dict

feat_data,label_dict = get_train_val_test()

# print('save label and feat')
# f = open('../tmp/label_dict_label','w')
# f.write(str(label_dict))
# f.close()
# feat_data.to_csv('../tmp/feat_data.csv',index=False)
#
# print('load cultures feat form file 901')
# feat_data = feat_data.reset_index(drop=True)
# tmp = pd.read_csv('../tmp/xxxxx.csv')
#
# feat_data = pd.concat([feat_data,tmp],axis=1)

print(feat_data.shape)
# del feat_data['video_seq_']
del feat_data['register_day']
print(feat_data.dtypes)
##################################################################

print('split val and train and test for all data',feat_data.shape)
sub = feat_data[feat_data['day']==30]
to_sub = feat_data[feat_data['day']!=30]
val = feat_data[feat_data['day'].isin([23])]
train = feat_data[~feat_data['day'].isin([23,30])]

# print(train['day'].unique())
# train_user_id = train[train['day']==16]['user_id'].unique()
# train_ext = train[train['day']!=16]
# train_ext = train_ext[~train_ext['user_id'].isin(train_user_id)]
# print('ext_feat',train_ext.shape)
# print('16_day_shape',train[train['day']==16].shape)
# train = pd.concat([train_ext,train[train['day']==16]])
# print('all_data',train.shape)


print('sample ratdio')
print(train[train['target']==1].shape[0],train.shape[0])
print(val[val['target']==1].shape[0],val.shape[0])
print(sub[sub['target']==1].shape[0],sub.shape[0])

print(sub['day'].unique(),sub.shape)
print(val['day'].unique(),val.shape)
print(train['day'].unique(),train.shape)

del sub['day'],val['day'],train['day'],to_sub['day']


y_train = train.pop('target')
train_user_index = train.pop('user_id')
X_train = train.values
print(train.columns)

y_test = val.pop('target')
val_user_index = val.pop('user_id')
X_test = val[train.columns].values

y_sub = sub.pop('target')
sub_user_index = sub.pop('user_id')
X_sub = sub[train.columns].values

y_to_sub = to_sub.pop('target')
to_sub_user_index = to_sub.pop('user_id')
X_to_sub = to_sub[train.columns].values

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_train_2 = lgb.Dataset(X_to_sub, y_to_sub)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
lgb_eval_2 = lgb.Dataset(X_to_sub, y_to_sub, reference=lgb_train_2)


# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'two_round':'true',
    'seed':42
}

print('Start training...')
# train

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=30000,
                valid_sets=lgb_eval,
                verbose_eval=250,
                early_stopping_rounds=250,
                )

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print('Feature importances:', list(gbm.feature_importance()))
imp = pd.DataFrame()
imp['col'] = list(train.columns)
imp['feat'] = list(gbm.feature_importance())
val_df = pd.DataFrame()
val_df['user_id'] = list(val_user_index.values)
val_df['y_pred'] = list(y_pred)
val_df = val_df.sort_values('y_pred',ascending=False)

select_dict = {}
for i in range(10):
    print((i + 1.0) / 10)
    select_dict[(i+1.0)/10] = get_score(set(val_df[val_df['y_pred']>(i+1.0)/10]['user_id']),label_dict[23],(i+1.0)/10)

select_dict_zip = zip(select_dict.keys(),select_dict.values())
select_dict_sort = sorted(select_dict_zip,key=lambda x:x[1],reverse=True)[0]
print('beat values',select_dict_sort[0])
print('beat score',round(select_dict_sort[1],6))

print(int(gbm.best_iteration * 1.1))
print(int(gbm.best_iteration))
print('predict_2')
gbm_2 = lgb.train(params,
                lgb_train_2,
                num_boost_round= int(gbm.best_iteration * 1.1),
                valid_sets=lgb_eval_2,
                verbose_eval=50,
                )

import datetime
submit = gbm_2.predict(X_sub, num_iteration=gbm_2.best_iteration)
sub_df = pd.DataFrame()
sub_df['user_id'] = list(sub_user_index.values)
sub_df['y_pred'] = list(submit)

print('sub_values',select_dict_sort[0])
sub_df = sub_df[sub_df['y_pred']>select_dict_sort[0]]['user_id']

sub_df = pd.DataFrame(sub_df).drop_duplicates()
sub_df.to_csv('../submit/%s_%s.csv'%(str(datetime.datetime.now().date()).replace('-',''),str(round(select_dict_sort[1],6)).split('.')[1]),index=False,header=None)

print('save model')
gbm.save_model('../model/model_%s_%s.txt'%(str(datetime.datetime.now().date()).replace('-',''),str(round(select_dict_sort[1],6)).split('.')[1]))

print(time.time()-start_time)

# 采取固定提交策略
