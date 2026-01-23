import os

def init_setting_KuaiSAR():
    global load_path, user_vocab, query_vocab, src_train, src_val, src_test, rec_train, rec_val, rec_test
    load_path = "../KuaiSAR/processed_data"
    user_vocab = os.path.join(load_path, 'user_vocab.pkl')
    query_vocab = os.path.join(load_path, 'query_vocab.pkl')

    src_train = os.path.join(load_path, 'src_train.pkl')
    src_val = os.path.join(load_path, 'src_val.pkl')
    src_test = os.path.join(load_path, 'src_test.pkl')

    rec_train = os.path.join(load_path, 'rec_train.pkl')
    rec_val = os.path.join(load_path, 'rec_val.pkl')
    rec_test = os.path.join(load_path, 'rec_test.pkl')

    global user_map_vocab, item_map_vocab, session_map_vocab
    user_map_vocab = os.path.join(load_path, 'user_feats_vocab.pkl')
    item_map_vocab = os.path.join(load_path, 'item_feats_vocab.pkl')
    session_map_vocab = os.path.join(load_path, 'search_vocab.pkl')

    global rec_inter_behavior, item_feature_list, item_text_feature
    rec_inter_behavior = ['click', 'like', 'follow', 'forward']
    item_feature_list = ['item_id', 'caption', 'first_level_category_id', 'second_level_category_id']
    item_text_feature = ['caption']

    global item_id_num, first_level_category_id_num, second_level_category_id_num, item_id_dim, first_level_category_id_dim, second_level_category_id_dim
    item_id_num = 552632 + 1
    first_level_category_id_num = 37 + 1
    second_level_category_id_num = 293 + 1
    item_id_dim = 32
    first_level_category_id_dim = 32
    second_level_category_id_dim = 32

    global user_feature_list, user_id_num, onehot_feat1_num, onehot_feat2_num, search_active_level_num, reco_active_level_num
    user_feature_list = ['user_id', 'onehot_feat1', 'onehot_feat2', 'search_active_level', 'reco_active_level']
    user_id_num = 18286
    onehot_feat1_num = 8
    onehot_feat2_num = 3
    search_active_level_num = 7
    reco_active_level_num = 4

    global user_id_dim, onehot_feat1_dim, onehot_feat2_dim, search_active_level_dim, reco_active_level_dim
    user_id_dim = 32
    onehot_feat1_dim = 32
    onehot_feat2_dim = 32
    search_active_level_dim = 32
    reco_active_level_dim = 32

    global word_id_num, word_id_dim, final_emb_size
    word_id_num = 677773 + 1
    word_id_dim = 32
    final_emb_size = 128

    global max_rec_his_len, max_src_session_his_len, max_his_len, max_session_item_len, max_query_word_len
    max_rec_his_len = 30
    max_src_session_his_len = 30
    max_his_len = 60
    max_session_item_len = 5
    max_query_word_len = 50

def init_setting_JDsearch():
    global load_path, user_vocab, query_vocab, src_train, src_val, src_test, rec_train, rec_val, rec_test
    load_path = "../JDsearch/processed_data"
    user_vocab = os.path.join(load_path, 'user_vocab.pkl')
    query_vocab = os.path.join(load_path, 'query_vocab.pkl')

    src_train = os.path.join(load_path, 'src_train.pkl')
    src_val = os.path.join(load_path, 'src_val.pkl')
    src_test = os.path.join(load_path, 'src_test.pkl')

    rec_train = os.path.join(load_path, 'rec_train.pkl')
    rec_val = os.path.join(load_path, 'rec_val.pkl')
    rec_test = os.path.join(load_path, 'rec_test.pkl')

    global user_map_vocab, item_map_vocab, session_map_vocab
    user_map_vocab = os.path.join(load_path, 'user_feats_vocab.pkl')
    item_map_vocab = os.path.join(load_path, 'item_feats_vocab.pkl')
    session_map_vocab = os.path.join(load_path, 'search_vocab.pkl')

    global rec_inter_behavior, item_feature_list, item_text_feature
    rec_inter_behavior = ['CLICK', 'ORD', 'CART', 'FLW']
    item_feature_list = ['item_id', 'caption', 'brand_id', 'cate_id_1', 'shop_id']
    item_text_feature = ['caption']

    global item_id_num, brand_id_num, cate_id_1_num, shop_id_num, item_id_num, item_id_dim, brand_id_dim, cate_id_1_dim, shop_id_dim
    item_id_num = 233341 + 1
    brand_id_num = 29423 + 1
    cate_id_1_num = 49 + 1
    shop_id_num = 45096 + 1
    item_id_dim = 32
    brand_id_dim = 32
    cate_id_1_dim = 32
    shop_id_dim = 32

    global user_feature_list, user_id_num, search_active_level_num, reco_active_level_num
    user_feature_list = ['user_id', 'search_active_level', 'reco_active_level']
    user_id_num = 35461
    search_active_level_num = 5
    reco_active_level_num = 5

    global user_id_dim, search_active_level_dim, reco_active_level_dim
    user_id_dim = 32
    search_active_level_dim = 32
    reco_active_level_dim = 32

    global word_id_num, word_id_dim, final_emb_size
    word_id_num = 146343 + 1
    word_id_dim = 32
    final_emb_size = 128

    global max_rec_his_len, max_src_session_his_len, max_his_len, max_session_item_len, max_query_word_len
    max_rec_his_len = 30
    max_src_session_his_len = 30
    max_his_len = 60
    max_session_item_len = 5
    max_query_word_len = 50