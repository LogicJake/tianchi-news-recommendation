import argparse
import os
import random
from random import sample

import pandas as pd
from tqdm import tqdm

from utils import Logger

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='数据处理')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'数据处理，mode: {mode}')


def data_offline(df_train_click, df_test_click):
    train_users = df_train_click['user_id'].values.tolist()
    # 随机采样出一部分样本
    val_users = sample(train_users, 50000)
    log.debug(f'val_users num: {len(set(val_users))}')

    # 训练集用户 抽出行为数据最后一条作为线下验证集
    click_list = []
    valid_query_list = []

    groups = df_train_click.groupby(['user_id'])
    for user_id, g in tqdm(groups):
        if user_id in val_users:
            valid_query = g.tail(1)
            valid_query_list.append(
                valid_query[['user_id', 'click_article_id']])

            train_click = g.head(g.shape[0] - 1)
            click_list.append(train_click)
        else:
            click_list.append(g)

    df_train_click = pd.concat(click_list, sort=False)
    df_valid_query = pd.concat(valid_query_list, sort=False)

    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = pd.concat([df_valid_query, df_test_query],
                         sort=False).reset_index(drop=True)
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../user_data/data/offline', exist_ok=True)

    df_click.to_pickle('../user_data/data/offline/click.pkl')
    df_query.to_pickle('../user_data/data/offline/query.pkl')


def data_online(df_train_click, df_test_click):
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    df_test_query = pd.DataFrame(test_query_list,
                                 columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_click],
                         sort=False).reset_index(drop=True)
    df_click = df_click.sort_values(['user_id',
                                     'click_timestamp']).reset_index(drop=True)

    log.debug(
        f'df_query shape: {df_query.shape}, df_click shape: {df_click.shape}')
    log.debug(f'{df_query.head()}')
    log.debug(f'{df_click.head()}')

    # 保存文件
    os.makedirs('../data/online', exist_ok=True)

    df_click.to_pickle('../user_data/data/online/click.pkl')
    df_query.to_pickle('../user_data/data/online/query.pkl')


if __name__ == '__main__':
    df_train_click = pd.read_csv('../tcdata/train_click_log.csv')
    df_test_click = pd.read_csv('../tcdata/testB_click_log_Test_B.csv')

    log.debug(
        f'df_train_click shape: {df_train_click.shape}, df_test_click shape: {df_test_click.shape}'
    )

    if mode == 'valid':
        data_offline(df_train_click, df_test_click)
    else:
        data_online(df_train_click, df_test_click)
