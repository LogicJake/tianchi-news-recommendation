import logging
import os
import pickle
import signal
from random import sample

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(
        self,
        filename,
        level='debug',
        fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    ):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))

        sh = logging.StreamHandler()
        sh.setFormatter(format_str)

        th = logging.FileHandler(filename=filename, encoding='utf-8', mode='a')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
                        np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def evaluate(df, total):
    hitrate_5 = 0
    mrr_5 = 0

    hitrate_10 = 0
    mrr_10 = 0

    hitrate_20 = 0
    mrr_20 = 0

    hitrate_40 = 0
    mrr_40 = 0

    hitrate_50 = 0
    mrr_50 = 0

    gg = df.groupby(['user_id'])

    for _, g in tqdm(gg):
        try:
            item_id = g[g['label'] == 1]['article_id'].values[0]
        except Exception as e:
            continue

        predictions = g['article_id'].values.tolist()

        rank = 0
        while predictions[rank] != item_id:
            rank += 1

        if rank < 5:
            mrr_5 += 1.0 / (rank + 1)
            hitrate_5 += 1

        if rank < 10:
            mrr_10 += 1.0 / (rank + 1)
            hitrate_10 += 1

        if rank < 20:
            mrr_20 += 1.0 / (rank + 1)
            hitrate_20 += 1

        if rank < 40:
            mrr_40 += 1.0 / (rank + 1)
            hitrate_40 += 1

        if rank < 50:
            mrr_50 += 1.0 / (rank + 1)
            hitrate_50 += 1

    hitrate_5 /= total
    mrr_5 /= total

    hitrate_10 /= total
    mrr_10 /= total

    hitrate_20 /= total
    mrr_20 /= total

    hitrate_40 /= total
    mrr_40 /= total

    hitrate_50 /= total
    mrr_50 /= total

    return hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50


@multitasking.task
def gen_sub_multitasking(test_users, prediction, all_articles, worker_id):
    lines = []

    for test_user in tqdm(test_users):
        g = prediction[prediction['user_id'] == test_user]
        g = g.head(5)
        items = g['article_id'].values.tolist()

        if len(set(items)) < 5:
            buchong = all_articles - set(items)
            buchong = sample(buchong, 5 - len(set(items)))
            items += buchong

        assert len(set(items)) == 5

        lines.append([test_user] + items)

    os.makedirs('../user_data/tmp/sub', exist_ok=True)

    with open(f'../user_data/tmp/sub/{worker_id}.pkl', 'wb') as f:
        pickle.dump(lines, f)


def gen_sub(prediction):
    prediction.sort_values(['user_id', 'pred'],
                           inplace=True,
                           ascending=[True, False])

    all_articles = set(prediction['article_id'].values)

    sub_sample = pd.read_csv('../tcdata/testB_click_log_Test_B.csv')
    test_users = sub_sample.user_id.unique()

    n_split = max_threads
    total = len(test_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/sub'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = test_users[i:i + n_len]
        gen_sub_multitasking(part_users, prediction, all_articles, i)

    multitasking.wait_for_tasks()

    lines = []
    for path, _, file_list in os.walk('../user_data/tmp/sub'):
        for file_name in file_list:
            with open(os.path.join(path, file_name), 'rb') as f:
                line = pickle.load(f)
                lines += line

    df_sub = pd.DataFrame(lines)
    df_sub.columns = [
        'user_id', 'article_1', 'article_2', 'article_3', 'article_4',
        'article_5'
    ]
    return df_sub
