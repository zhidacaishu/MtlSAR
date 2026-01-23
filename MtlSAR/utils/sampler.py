import numpy as np
import pandas as pd
from models import const

class Sampler(object):
    def __init__(self, data_path, search, user_vocab):
        self.src = search
        self.user_vocab = user_vocab
        self.data = pd.read_pickle(data_path)

    def sample(self, index):
        feed_dict = {}
        line = self.data.iloc[index]

        user = int(line['user_id'])
        feed_dict['user'] = [user]
        feed_dict['item'] = [int(line['item_id'])]
        feed_dict['neg_items'] = [line['neg_items']]
        feed_dict['src'] = self.src

        if self.src:
            query = self.get_pad_query(line['keyword'])
            feed_dict['query'] = list([query])
            feed_dict['weak_neg_items'] = [line['weak_neg_items']]
        else:
            feed_dict['behavior_type'] = [line['behavior_type']]
        
        rec_his_num = int(line['rec_his_num'])
        src_session_his_num = int(line['src_session_his_num'])

        rec_his_item = self.user_vocab[user]['rec_his'][:rec_his_num][-const.max_rec_his_len:]
        rec_his_ts = self.user_vocab[user]['rec_his_ts'][:rec_his_num][-const.max_rec_his_len:]
        rec_his_type = self.user_vocab[user]['rec_his_type'][:rec_his_num][-const.max_rec_his_len:]
        if len(rec_his_item) < const.max_rec_his_len:
            rec_his_item += [0] * (const.max_rec_his_len - len(rec_his_item))
            rec_his_ts += [np.inf] * (const.max_rec_his_len - len(rec_his_ts))
            rec_his_type += [[0] * (len(const.rec_inter_behavior) + 1)] * (const.max_rec_his_len - len(rec_his_type))
        rec_his = list(zip(rec_his_item, rec_his_ts, rec_his_type))

        src_his_item = self.user_vocab[user]['src_session_his'][:src_session_his_num][-const.max_src_session_his_len:]
        src_his_ts = self.user_vocab[user]['src_session_his_ts'][:src_session_his_num][-const.max_src_session_his_len:]
        src_his_type = [[0, 0, 0, 0, 1]] * len(src_his_item)
        if len(src_his_item) < const.max_src_session_his_len:
            src_his_item += [0] * (const.max_src_session_his_len - len(src_his_item))
            src_his_ts += [np.inf] * (const.max_src_session_his_len - len(src_his_ts))
            src_his_type += [[0] * (len(const.rec_inter_behavior) + 1)] * (const.max_src_session_his_len - len(src_his_type))
        src_his = list(zip(src_his_item, src_his_ts, src_his_type))

        all_his = rec_his + src_his
        sorted_all_his = sorted(all_his, key=lambda x: x[1])
        sorted_all_his_item = [x[0] for x in sorted_all_his]
        sorted_all_his_time = [x[1] for x in sorted_all_his]
        sorted_all_his_type = [x[2] for x in sorted_all_his]

        feed_dict['all_his'] = [sorted_all_his_item]
        feed_dict['all_his_ts'] = [sorted_all_his_time]
        feed_dict['all_his_type'] = [sorted_all_his_type]

        return feed_dict
    
    def get_pad_query(self, query):
        if type(query) == str:
            query = eval(query)
        elif type(query) == int:
            query = [query]
        query = query[:const.max_query_word_len]
        if len(query) < const.max_query_word_len:
            query += [0] * (const.max_query_word_len - len(query))
        return query

