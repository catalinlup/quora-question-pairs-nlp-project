from utils import save_feature, load_feature
from dataset import *
import numpy as np
from collections import Counter
import numpy as np
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
from bert_similarity import cosine_similarity
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer



DELTA=1e-5


def compute_common_words(q1_tokenized, q2_tokenized):
    common_words = set(q1_tokenized + q2_tokenized)
    return float(len(common_words))

def compute_common_words_vec(question_set_tokenized):
    question1 = question_set_tokenized[0]
    question2 = question_set_tokenized[1]
    return np.array([compute_common_words(q1, q2) for q1, q2 in zip(question1, question2)]).astype(np.float32)
    

def compute_total_words_vec(question_set_tokenized):
    question1 = question_set_tokenized[0]
    question2 = question_set_tokenized[1]
    return np.array([len(q1) + len(q2) for q1, q2 in zip(question1, question2)]).astype(np.float32)


def compute_q1_n_words_vec(question_set_tokenized):
    question1 = question_set_tokenized[0]
    return np.array([len(q1) for q1 in question1]).astype(np.float32)

def compute_q2_n_words_vec(question_set_tokenized):
    question2 = question_set_tokenized[1]
    return np.array([len(q2) for q2 in question2]).astype(np.float32)

def compute_q1len_vec(question_set):
    question1 = question_set[0]
    return np.array([len(q1) for q1 in question1]).astype(np.float32)

def compute_q2len_vec(question_set):
    question2 = question_set[1]
    return np.array([len(q2) for q2 in question2]).astype(np.float32)


def compute_freq_qid(data_id):
    q1_id = data_id[0]
    q2_id = data_id[1]
    unique, counts = np.unique(q1_id + q2_id, return_counts=True)

    table = dict()
    for u, c in zip(unique, counts):
        table[u] = c

    freq_qid1 = np.array([table[id] for id in q1_id])
    freq_qid2 = np.array([table[id] for id in q2_id])

    return freq_qid1, freq_qid2


def compute_last_word_eq(question_set_tokenized):
    question1 = question_set_tokenized[0]
    question2 = question_set_tokenized[1]

    features = []

    for q1, q2 in zip(question1, question2):
        if len(q1) < 1 or len(q2) < 1:
            features.append(0)
            continue
        features.append(q1[-1] == q2[-1])

    return np.array(features).astype(np.float32)

def compute_first_word_eq(question_set_tokenized):
    question1 = question_set_tokenized[0]
    question2 = question_set_tokenized[1]

    features = []

    for q1, q2 in zip(question1, question2):
        if len(q1) < 1 or len(q2) < 1:
            features.append(0)
            continue
        features.append(q1[0] == q2[0])

    return np.array(features).astype(np.float32)


def fuzz_ratio(s1, s2):
    return fuzz.ratio(s1.lower(), s2.lower()) / 100

def compute_fuzz_ratio(question_set):
    question1 = question_set[0]
    question2 = question_set[1]
    return np.array([fuzz_ratio(q1, q2) for q1, q2 in zip(question1, question2)]).astype(np.float32)

def fuzz_partial_ratio(s1, s2):
    return fuzz.partial_ratio(s1.lower(), s2.lower()) / 100

def compute_fuzz_partial_ratio(question_set):
    question1 = question_set[0]
    question2 = question_set[1]
    return np.array([fuzz_partial_ratio(q1, q2) for q1, q2 in zip(question1, question2)]).astype(np.float32)

def fuzz_token_sort_ratio(s1, s2):
    return fuzz.token_sort_ratio(s1.lower(), s2.lower()) / 100


def compute_fuzz_token_sort_ratio(question_set):
    question1 = question_set[0]
    question2 = question_set[1]
    return np.array([fuzz_token_sort_ratio(q1, q2) for q1, q2 in zip(question1, question2)]).astype(np.float32)


def fuzz_token_set_ratio(s1, s2):
    return fuzz.token_set_ratio(s1.lower(), s2.lower()) / 100




def sentance_similarity(tokens1, tokens2, emb, stp):
    arr1 = [emb[t1]  for t1 in tokens1 if (t1 in emb.keys()) and not (t1 in stp)]
    arr2 = [emb[t2]  for t2 in tokens2 if (t2 in emb.keys()) and not (t2 in stp)]

    if len(arr1) == 0 or len(arr2) == 0:
        return 0.0

    avg_embedding1 = np.sum(np.stack(arr1), axis=0)
    avg_embedding1 = avg_embedding1 / np.linalg.norm(avg_embedding1)
    avg_embedding2 = np.sum(np.stack(arr2), axis=0)
    avg_embedding2 = avg_embedding2 / np.linalg.norm(avg_embedding2)

    # print(avg_embedding1.shape)

    return np.sum(avg_embedding1 * avg_embedding2) / (np.linalg.norm(avg_embedding1) * np.linalg.norm(avg_embedding2) + DELTA)

def sentance_similarity_euclidean(tokens1, tokens2, emb, stp):
    arr1 = [emb[t1]  for t1 in tokens1 if (t1 in emb.keys()) and not (t1 in stp)]
    arr2 = [emb[t2]  for t2 in tokens2 if (t2 in emb.keys()) and not (t2 in stp)]

    if len(arr1) == 0 or len(arr2) == 0:
        return 0.0

    avg_embedding1 = np.sum(np.stack(arr1), axis=0)
    avg_embedding1 = avg_embedding1 / np.linalg.norm(avg_embedding1)
    avg_embedding2 = np.sum(np.stack(arr2), axis=0)
    avg_embedding2 = avg_embedding2 / np.linalg.norm(avg_embedding2)

    # print(avg_embedding1.shape)

    return np.mean((avg_embedding1 - avg_embedding2) ** 2)

def sentance_similarity_concat(tokens1, tokens2, emb, stp):
    arr1 = [emb[t1]  for t1 in tokens1 if (t1 in emb.keys()) and not (t1 in stp)]
    arr2 = [emb[t2]  for t2 in tokens2 if (t2 in emb.keys()) and not (t2 in stp)]

    if len(arr1) == 0 or len(arr2) == 0:
        return np.zeros(600)

    avg_embedding1 = np.sum(np.stack(arr1), axis=0)
    avg_embedding1 = avg_embedding1 / np.linalg.norm(avg_embedding1)
    avg_embedding2 = np.sum(np.stack(arr2), axis=0)
    avg_embedding2 = avg_embedding2 / np.linalg.norm(avg_embedding2)

    # print(avg_embedding1.shape)

    return np.concatenate([avg_embedding1, avg_embedding2], axis=0)


def compute_sentance_similarity(question_tokenized_set):
    emb = load_feature('glove_300d')
    question1 = question_tokenized_set[0]
    question2 = question_tokenized_set[1]
    stp = stopwords.words('english')

   


    return np.array([sentance_similarity(q1, q2, emb, stp) for q1, q2, index in zip(question1, question2, range(len(question1)))]).astype(np.float32)


def compute_sentance_similarity_euclidean(question_tokenized_set):
    emb = load_feature('glove_300d')
    question1 = question_tokenized_set[0]
    question2 = question_tokenized_set[1]
    stp = stopwords.words('english')

   
    return np.array([sentance_similarity_euclidean(q1, q2, emb, stp) for q1, q2, index in zip(question1, question2, range(len(question1)))]).astype(np.float32)


def create_sentance_embeddings_feature_matrix(question_tokenized_set):
    emb = load_feature('glove_300d')
    question1 = question_tokenized_set[0]
    question2 = question_tokenized_set[1]
    stp = stopwords.words('english')

   
    return np.stack([sentance_similarity_concat(q1, q2, emb, stp) for q1, q2, index in zip(question1, question2, range(len(question1)))]).astype(np.float32)




def compute_fuzz_token_set_ratio(question_set):
    question1 = question_set[0]
    question2 = question_set[1]
    return np.array([fuzz_token_set_ratio(q1, q2) for q1, q2 in zip(question1, question2)]).astype(np.float32)


def compute_transformers(question_set):
    model = SentenceTransformer('stsb-roberta-large')
    embedding1 = model.encode(question_set[0], convert_to_tensor=True)
    embedding2 = model.encode(question_set[1], convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)

    return np.array([cosine_scores[i][i] for i in range(len(question_set[0]))])


def compute_basic_features(data_tokenized, data, data_id, name):
    common_words = compute_common_words_vec(data_tokenized)
    save_feature(common_words, f'{name}_common_words')

    total_words = compute_total_words_vec(data_tokenized)
    save_feature(total_words, f'{name}_total_words')

    q1_n_words = compute_q1_n_words_vec(data_tokenized)
    save_feature(q1_n_words, f'{name}_q1_n_words')

    q2_n_words = compute_q2_n_words_vec(data_tokenized)
    save_feature(q2_n_words, f'{name}_q2_n_words')

    q1len = compute_q1len_vec(data)
    save_feature(q1len, f'{name}_q1len')

    q2len = compute_q2len_vec(data)
    save_feature(q2len, f'{name}_q2len')

    freq_qid1, freq_qid2 = compute_freq_qid(data_id)
    save_feature(freq_qid1, f'{name}_freq_qid1')
    save_feature(freq_qid2, f'{name}_freq_qid2')
    save_feature(freq_qid1 + freq_qid2, f'{name}_freq_sum')
    save_feature(freq_qid1 - freq_qid2, f'{name}_freq_diff')

    cwc_min = common_words / (np.min(np.stack([q1_n_words, q2_n_words], axis=1)) + DELTA)
    save_feature(cwc_min, f'{name}_cwc_min')

    cwc_max = common_words / (np.max(np.stack([q1_n_words, q2_n_words], axis=1)) + DELTA)
    save_feature(cwc_max, f'{name}_cwc_max')

    abs_len_diff = np.abs(q1len - q2len)
    save_feature(abs_len_diff, f'{name}_abs_len_diff')

    first_word_eq = compute_first_word_eq(data_tokenized)
    save_feature(first_word_eq, f'{name}_first_word_eq')

    last_word_eq = compute_last_word_eq(data_tokenized)
    save_feature(last_word_eq, f'{name}_last_word_eq')

    # fuzzy features
    fuzz_ratio = compute_fuzz_ratio(data)
    save_feature(fuzz_ratio, f'{name}_fuzz_ratio')

    fuzz_partial_ratio = compute_fuzz_partial_ratio(data)
    save_feature(fuzz_partial_ratio, f'{name}_fuzz_partial_ratio')

    fuzz_sort_ratio = compute_fuzz_token_sort_ratio(data)
    save_feature(fuzz_sort_ratio, f'{name}_fuzz_sort_ratio')

    fuzz_set_ratio = compute_fuzz_token_set_ratio(data)
    save_feature(fuzz_set_ratio, f'{name}_fuzz_set_ratio')

    sentance_similarity = compute_sentance_similarity(data_tokenized)
    save_feature(sentance_similarity, f'{name}_sim')

    sentance_similarity_euclidean = compute_sentance_similarity_euclidean(data_tokenized)
    save_feature(sentance_similarity_euclidean, f'{name}_sim')

    emb_concat = create_sentance_embeddings_feature_matrix(data_tokenized)
    save_feature(emb_concat, f'{name}_emb_concat')

    # transformer_similarity = compute_transformers(data)
    # save_feature(transformer_similarity, f'{name}_transf')


    full_feature_map = np.stack([
        common_words, total_words, q1_n_words, 
        q2_n_words, q1len, q2len, freq_qid1, freq_qid2,
        cwc_min, cwc_max, abs_len_diff, first_word_eq,
        last_word_eq, fuzz_ratio, fuzz_partial_ratio, fuzz_sort_ratio, fuzz_set_ratio, sentance_similarity,sentance_similarity_euclidean
        ], axis=1)
    
    header_names = [
        "common_words", "total_words", "q1_n_words", 
        "q2_n_words", "q1len", "q2len", "freq_qid1", "freq_qid2",
        "cwc_min", "cwc_max", "abs_len_diff", "first_word_eq",
        "last_word_eq", "fuzz_ratio", "fuzz_partial_ratio", "fuzz_sort_ratio", "fuzz_set_ratio", "sentance_similarity", "sentance_similarity_euclidean"
        ]
    
    features_df = pd.DataFrame(full_feature_map, columns=header_names)
    features_df.to_csv(f'features/{name}_feature_map.csv')

    
    




compute_basic_features(question_validation_set_tokenized, question_validation_set, validation_data_id, 'validation')
compute_basic_features(question_set_tokenized, question_set, data_id, 'train_test')



# print(fuzz_ratio("Cool", "Coul"))