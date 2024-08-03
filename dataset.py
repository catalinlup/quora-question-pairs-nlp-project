import pandas as pd
from global_variables import *
# from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

def tokenize(sentances):
    return [tokenizer.tokenize(s) for s in sentances]

data = pd.read_csv(TRAINING_SET_PATH).dropna()
data_id = [data['qid1'].to_list(), data['qid2'].to_list()]
question1 = data["question1"].to_list()
question1_tokenized = tokenize(question1)
question2 = data["question2"].to_list()
question2_tokenized = tokenize(question2)
question_set = [question1, question2]
question_set_tokenized = [question1_tokenized, question2_tokenized]


# LOAD THE VALIDATION DATA
validation_data = pd.read_csv(TEST_SET_PATH)
validation_data_id = [validation_data['qid1'].to_list(), validation_data['qid2'].to_list()]
question1_validation = validation_data["question1"].to_list()
question1_validation_tokenized = tokenize(question1_validation)
question2_validation = validation_data["question2"].to_list()
question2_validation_tokenized = tokenize(question2_validation)
question_validation_set = [question1_validation, question2_validation]
question_validation_set_tokenized = [question1_validation_tokenized, question2_validation_tokenized]


