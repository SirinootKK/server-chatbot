import random
import time
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer, util
from pythainlp import Tokenizer
import pickle
# import evaluate

# class SemanticModel:
#     SHEET_NAME_MDEBERTA = 'mdeberta'
#     SHEET_NAME_DEFAULT = 'Default'
#     UNKNOWN_ANSWERS = ["กรุณาลงรายระเอียดมากกว่านี้ได้มั้ยคะ", "ขอโทษค่ะลูกค้า ดิฉันไม่ทราบจริง ๆ"]

#     def __init__(self, df_path=None, test_df_path=None, model_path=None, tokenizer_path=None, embedding_model_name=None, embeddingsPath=None):
#         self.df = None
#         self.test_df = None
#         self.model = None
#         self.tokenizer = None
#         self.embedding_model = None
#         self.index = None
#         self.k = 1
#         if all(arg is not None for arg in (df_path, test_df_path, model_path, tokenizer_path, embedding_model_name, embeddingsPath)):
#             self.set_df(df_path)
#             self.set_test_df(test_df_path)
#             self.set_model(model_path)
#             self.set_tokenizer(tokenizer_path)
#             self.set_embedding_model(embedding_model_name)
#             self.set_index(self.prepare_sentences_vector(self.load_embeddings(embeddingsPath)))

#     def set_df(self, path):
#         self.df = pd.read_excel(path, sheet_name='Default')
#         self.df.rename(columns={'Response': 'Answer'}, inplace=True)
#         self.df['context'] = pd.read_excel(path, sheet_name='mdeberta')['context']

#     def set_test_df(self, path):
#         self.test_df = pd.read_excel(path, sheet_name='Test')

#     def set_model(self, model):
#         self.model = AutoModelForQuestionAnswering.from_pretrained(model)

#     def set_tokenizer(self, tokenizer):
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

#     def set_embedding_model(self, model):
#         self.embedding_model = SentenceTransformer(model)

#     def set_index(self, vector):
#         if torch.cuda.is_available():  # Check if GPU is available
#             res = faiss.StandardGpuResources()
#             self.index = faiss.IndexFlatL2(vector.shape[1])
#             gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, self.index)
#             gpu_index_flat.add(vector)
#             self.index = gpu_index_flat
#         else:  # If GPU is not available, use CPU-based Faiss index
#             self.index = faiss.IndexFlatL2(vector.shape[1])
#             self.index.add(vector)

#     def set_k(self, k_value):
#         self.k = k_value

#     def get_df(self):
#         return self.df

#     def get_test_df(self):
#         return self.test_df

#     def get_model(self):
#         return self.model

#     def get_tokenizer(self):
#         return self.tokenizer

#     def get_embedding_model(self):
#         return self.embedding_model

#     def get_index(self):
#         return self.index

#     def get_k(self):
#         return self.k

#     def get_embeddings(self, text_list):
#         return self.embedding_model.encode(text_list)

#     def prepare_sentences_vector(self, encoded_list):
#         encoded_list = [i.reshape(1, -1) for i in encoded_list]
#         encoded_list = np.vstack(encoded_list).astype('float32')
#         encoded_list = normalize(encoded_list)
#         return encoded_list

#     def store_embeddings(self, embeddings):
#         with open('embeddings.pkl', "wb") as fOut:
#             pickle.dump({'sentences': self.df['Question'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#     def load_embeddings(self, file_path):
#         with open(file_path, "rb") as fIn:
#             stored_data = pickle.load(fIn)
#             stored_sentences = stored_data['sentences']
#             stored_embeddings = stored_data['embeddings']
#         return stored_embeddings

#     def model_pipeline(self, question, similar_context):
#         inputs = self.tokenizer(question, similar_context, return_tensors="pt")
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         answer_start_index = outputs.start_logits.argmax()
#         answer_end_index = outputs.end_logits.argmax()
#         predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
#         Answer = self.tokenizer.decode(predict_answer_tokens)
#         return Answer
    
#     def faiss_search(self, question_vector):
#         distances, indices = self.index.search(question_vector, self.k)
#         similar_questions = [self.df['Question'][indices[0][i]] for i in range(self.k)]
#         similar_contexts = [self.df['context'][indices[0][i]] for i in range(self.k)]
#         return similar_questions, similar_contexts, distances, indices
    
#     def predict_bert_embedding(self,message):
#         message = message.strip()
#         question_vector = self.get_embeddings(message)
#         question_vector=self.prepare_sentences_vector([question_vector])
#         similar_questions, similar_contexts, distances,indices = self.faiss_search(question_vector)


#         Answer = self.model_pipeline(similar_questions, similar_contexts)

#         score = [distances[0][i] for i in range(self.k)]
#         output = {
#             "user_question": message,
#             "distances": [distances[0][i] for i in range(self.k)],
#             "similar_questions": [question.strip() for question in similar_questions],
#             "answer": Answer,
#         }
#         print('Predict using Faiss and Model Done')

#         return Answer, score

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import pickle

class SemanticModel:
    SHEET_NAME_MDEBERTA = 'mdeberta'
    SHEET_NAME_DEFAULT = 'Default'
    UNKNOWN_ANSWERS = ["กรุณาลงรายระเอียดมากกว่านี้ได้มั้ยคะ", "ขอโทษค่ะลูกค้า ดิฉันไม่ทราบจริง ๆ"]

    def __init__(self, df_path=None, test_df_path=None, model_path=None, tokenizer_path=None, embedding_model_name=None, embeddingsPath=None):
        self.df = None
        self.test_df = None
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.index = None
        self.k = 5
        if all(arg is not None for arg in (df_path, test_df_path, model_path, tokenizer_path, embedding_model_name, embeddingsPath)):
            self.set_df(df_path)
            self.set_test_df(test_df_path)
            self.set_model(model_path)
            self.set_tokenizer(tokenizer_path)
            self.set_embedding_model(embedding_model_name)
            self.set_index(self.prepare_sentences_vector(self.load_embeddings(embeddingsPath)))

    def set_df(self, path):
        self.df = pd.read_excel(path, sheet_name=self.SHEET_NAME_DEFAULT)
        self.df.rename(columns={'Response': 'Answer'}, inplace=True)
        self.df['context'] = pd.read_excel(path, self.SHEET_NAME_MDEBERTA)['context']

    def set_test_df(self, path):
        self.test_df = pd.read_excel(path, sheet_name='Test')

    def set_model(self, model):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def set_embedding_model(self, model):
        self.embedding_model = SentenceTransformer(model)

    def set_index(self, vector):
        if torch.cuda.is_available():  # Check if GPU is available
            res = faiss.StandardGpuResources()
            self.index = faiss.IndexFlatL2(vector.shape[1])
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, self.index)
            gpu_index_flat.add(vector)
            self.index = gpu_index_flat
        else:  # If GPU is not available, use CPU-based Faiss index
            self.index = faiss.IndexFlatL2(vector.shape[1])
            self.index.add(vector)

    def set_k(self, k_value):
        self.k = k_value

    def get_df(self):
        return self.df

    def get_test_df(self):
        return self.test_df

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_embedding_model(self):
        return self.embedding_model

    def get_index(self):
        return self.index

    def get_k(self):
        return self.k

    def get_embeddings(self, text_list):
        return self.embedding_model.encode(text_list)

    def prepare_sentences_vector(self, encoded_list):
        encoded_list = [i.reshape(1, -1) for i in encoded_list]
        encoded_list = np.vstack(encoded_list).astype('float32')
        encoded_list = normalize(encoded_list)
        return encoded_list

    def store_embeddings(self, embeddings):
        with open('embeddings.pkl', "wb") as fOut:
            pickle.dump({'sentences': self.df['Question'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    def load_embeddings(self, file_path):
        with open(file_path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
        return stored_embeddings

    def model_pipeline(self, question, similar_context):
        inputs = self.tokenizer(question, similar_context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        Answer = self.tokenizer.decode(predict_answer_tokens)
        return Answer
    
    def faiss_search(self, question_vector):
        distances, indices = self.index.search(question_vector, self.k)
        similar_questions = [self.df['Question'][indices[0][i]] for i in range(self.k)]
        similar_contexts = [self.df['context'][indices[0][i]] for i in range(self.k)]
        return similar_questions, similar_contexts, distances, indices
    
    def predict_bert_embedding(self, message):
        list_context_for_show = []
        list_distance_for_show = []
        list_similar_question = []

        question_vector = self.get_embeddings(message)
        question_vector = self.prepare_sentences_vector([question_vector])
        similar_questions, similar_contexts, distances, indices = self.faiss_search(question_vector)

        for i in range(min(5, self.k)):
            index = indices[0][i]
            similar_question = self.df['Question'][index]
            similar_context = self.df['context'][index]

            list_similar_question.append(similar_question)
            list_context_for_show.append(similar_context)
            list_distance_for_show.append(str(1 - distances[0][i]))

        distance = list_distance_for_show[0]  # Update this line to use the correct distance

        if float(distance) < 0.5:
            Answer = random.choice(self.UNKNOWN_ANSWERS)
        else:
            Answer = self.model_pipeline(list_similar_question[0], list_context_for_show[0])
            Answer = Answer.strip().replace("<unk>", "@")
            # print(Answer)

        return Answer, list_context_for_show, distance, list_distance_for_show


