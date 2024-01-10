from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from gensim.models.doc2vec import Doc2Vec
from pythainlp.tokenize import word_tokenize
from pythainlp.util import dict_trie
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import faiss
import torch
import torch.nn.functional as F
import random

# class QADoc2VecModel:
#     SHEET_NAME_MDEBERTA = 'mdeberta'
#     SHEET_NAME_DEFAULT = 'Default'
#     UNKNOWN_ANSWERS = ["กรุณาลงรายระเอียดมากกว่านี้ได้มั้ยคะ", "ขอโทษค่ะลูกค้า ดิฉันไม่ทราบจริง ๆ"]
    
#     def __init__(self, model_path, tokenizer_path, doc2vec_model_path, dataset_path, test_size=0.2, random_state=42):
#         self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         self.model_doc2vec = Doc2Vec.load(doc2vec_model_path)
#         self.df = pd.read_excel(dataset_path, sheet_name=self.SHEET_NAME_MDEBERTA)
#         _df = pd.read_excel(dataset_path, sheet_name=self.SHEET_NAME_DEFAULT)
#         self.df['answers'] = _df['Answer']
        
#         self.questions_vectors = self._prepare_sentence_vectors(self.df['question'])
#         self.ques_vec_norm = np.vstack(self.questions_vectors).astype('float32')
#         self.ques_vec_norm = normalize(self.ques_vec_norm)
        
#         self.context_vectors = self._prepare_sentence_vectors(self.df['context'])
#         self.context_vectors_for_faiss = np.vstack(self.context_vectors).astype('float32')
#         self.context_vectors_for_faiss = normalize(self.context_vectors_for_faiss)

#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             self.df['question'], self.df['answers'], test_size=test_size, random_state=random_state
#         )

#     def _tokenize_doc2vec(self, string):
#         unwanted_strings = {',' , ';'}
#         tokenized_list = []
#         tokenize_with_customdict = word_tokenize(string.upper())
#         for token in tokenize_with_customdict:
#             if not token.isspace():
#                 for s in unwanted_strings:
#                     token = token.replace(s, '')
#                 tokenized_list.append(word_tokenize(token, engine="newmm"))
#         tokenized_list = list(np.concatenate(tokenized_list))
#         tokenized_list = [elem for elem in tokenized_list if elem.strip()]
#         return tokenized_list

#     def _prepare_sentence_vectors(self, df_column):
#         context_vectors = []
#         for c in df_column:
#             self.model_doc2vec.random.seed(self.model_doc2vec.seed)
#             c_vector = self.model_doc2vec.infer_vector(self._tokenize_doc2vec(str(c))).reshape(1, -1)
#             context_vectors.append(c_vector)
#         return context_vectors

#     def _predict_model(self, question, similar_context):
#         inputs = self.tokenizer(question, similar_context, return_tensors="pt")
#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         start_logits = outputs.start_logits
#         end_logits = outputs.end_logits

#         start_probs = F.softmax(start_logits, dim=1)
#         end_probs = F.softmax(end_logits, dim=1)

#         answer_start_index = start_probs.argmax()
#         answer_end_index = end_probs.argmax()

#         predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
#         Answer = self.tokenizer.decode(predict_answer_tokens)
        
#         return Answer

#     def predict(self, message):
#         list_context_for_show = []
#         list_distance_for_show = []
        
#         index = faiss.IndexFlatL2(self.ques_vec_norm.shape[1])
#         index.add(self.ques_vec_norm)
#         message = message.strip(" \t\n")

#         self.model_doc2vec.random.seed(self.model_doc2vec.seed)
#         question_vector = self.model_doc2vec.infer_vector(self._tokenize_doc2vec(message)).reshape(1, -1)

#         if np.isnan(question_vector).any() or np.isinf(question_vector).any():
#             return "Unable to process the input. Please try again.", [], "N/A", []

#         question_vector = normalize(question_vector)
#         question_vector = question_vector.reshape(1, -1).astype('float32')
#         k = 5 
#         distances, indices = index.search(question_vector, k)
#         for i in range(min(5, k)):
#             similar_context_for_show = indices[0][i]
#             similar_context = self.df['context'][similar_context_for_show]
#             list_context_for_show.append(similar_context)
#             list_distance_for_show.append(str(1 - distances[0][i]))

#         distance = str(1 - distances[0][0])

#         similar_question_index = indices[0][0]
#         similar_question = self.df['question'][similar_question_index]
#         similar_context = self.df['context'][similar_question_index]

#         if float(distance) < 0.5:
#             Answer = random.choice(self.UNKNOWN_ANSWERS)
#             return Answer, list_context_for_show, distance, list_distance_for_show
#         else:
#             Answer = self._predict_model(similar_question, similar_context)
#             Answer = Answer.strip().replace("<unk>", "@")

#             return Answer, list_context_for_show, distance, list_distance_for_show
        
class QADoc2VecModel:
    SHEET_NAME_MDEBERTA = 'mdeberta'
    SHEET_NAME_DEFAULT = 'Default'
    UNKNOWN_ANSWERS = ["กรุณาลงรายระเอียดมากกว่านี้ได้มั้ยคะ", "ขอโทษค่ะลูกค้า ดิฉันไม่ทราบจริง ๆ"]

    def __init__(self, model_path=None, tokenizer_path=None, doc2vec_model_path=None, dataset_path=None, test_size=0.2, random_state=42):
        self.model = None
        self.tokenizer = None
        self.model_doc2vec = None
        self.df = None
        self.questions_vectors = None
        self.ques_vec_norm = None
        self.context_vectors = None
        self.context_vectors_for_faiss = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Initialization using setters
        if all(arg is not None for arg in (model_path, tokenizer_path, doc2vec_model_path, dataset_path)):
            self.set_model_path(model_path)
            self.set_tokenizer_path(tokenizer_path)
            self.set_doc2vec_model_path(doc2vec_model_path)
            self.set_dataset_path(dataset_path)

            # Additional setup
            self._setup(test_size, random_state)

    # Setters
    def set_model_path(self, model_path):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    def set_tokenizer_path(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def set_doc2vec_model_path(self, doc2vec_model_path):
        self.model_doc2vec = Doc2Vec.load(doc2vec_model_path)

    def set_dataset_path(self, dataset_path):
        self.df = pd.read_excel(dataset_path, sheet_name=self.SHEET_NAME_MDEBERTA)
        _df = pd.read_excel(dataset_path, sheet_name=self.SHEET_NAME_DEFAULT)
        self.df['answers'] = _df['Answer']

    # Additional setup method
    def _setup(self, test_size, random_state):
        self.questions_vectors = self._prepare_sentence_vectors(self.df['question'])
        self.ques_vec_norm = np.vstack(self.questions_vectors).astype('float32')
        self.ques_vec_norm = normalize(self.ques_vec_norm)

        self.context_vectors = self._prepare_sentence_vectors(self.df['context'])
        self.context_vectors_for_faiss = np.vstack(self.context_vectors).astype('float32')
        self.context_vectors_for_faiss = normalize(self.context_vectors_for_faiss)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df['question'], self.df['answers'], test_size=test_size, random_state=random_state
        )

    # Getter methods
    def get_model_path(self):
        return self.model.config.architectures[0]

    def get_tokenizer_path(self):
        return self.tokenizer.name_or_path

    def get_doc2vec_model_path(self):
        return self.model_doc2vec

    def get_dataset_path(self):
        return self.df

    # Helper methods
    def _tokenize_doc2vec(self, string):
        unwanted_strings = {',' , ';'}
        tokenized_list = []
        tokenize_with_customdict = word_tokenize(string.upper())
        for token in tokenize_with_customdict:
            if not token.isspace():
                for s in unwanted_strings:
                    token = token.replace(s, '')
                tokenized_list.append(word_tokenize(token))
        tokenized_list = list(np.concatenate(tokenized_list))
        tokenized_list = [elem for elem in tokenized_list if elem.strip()]
        return tokenized_list

    def _prepare_sentence_vectors(self, df_column):
        context_vectors = []
        for c in df_column:
            self.model_doc2vec.random.seed(self.model_doc2vec.seed)
            c_vector = self.model_doc2vec.infer_vector(self._tokenize_doc2vec(str(c))).reshape(1, -1)
            context_vectors.append(c_vector)
        return context_vectors

    def _predict_model(self, question, similar_context):
        inputs = self.tokenizer(question, similar_context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)

        answer_start_index = start_probs.argmax()
        answer_end_index = end_probs.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        Answer = self.tokenizer.decode(predict_answer_tokens)

        return Answer

    def predict(self, message):
        list_context_for_show = []
        list_distance_for_show = []

        index = faiss.IndexFlatL2(self.ques_vec_norm.shape[1])
        index.add(self.ques_vec_norm)
        message = message.strip(" \t\n")

        self.model_doc2vec.random.seed(self.model_doc2vec.seed)
        question_vector = self.model_doc2vec.infer_vector(self._tokenize_doc2vec(message)).reshape(1, -1)

        if np.isnan(question_vector).any() or np.isinf(question_vector).any():
            return "Unable to process the input. Please try again.", [], "N/A", []

        question_vector = normalize(question_vector)
        question_vector = question_vector.reshape(1, -1).astype('float32')
        k = 5
        distances, indices = index.search(question_vector, k)
        for i in range(min(5, k)):
            similar_context_for_show = indices[0][i]
            similar_context = self.df['context'][similar_context_for_show]
            list_context_for_show.append(similar_context)
            list_distance_for_show.append(str(1 - distances[0][i]))

        distance = str(1 - distances[0][0])

        similar_question_index = indices[0][0]
        similar_question = self.df['question'][similar_question_index]
        similar_context = self.df['context'][similar_question_index]

        if float(distance) < 0.5:
            Answer = random.choice(self.UNKNOWN_ANSWERS)
            return Answer, list_context_for_show, distance, list_distance_for_show
        else:
            Answer = self._predict_model(similar_question, similar_context)
            Answer = Answer.strip().replace("<unk>", "@")

            return Answer, list_context_for_show, distance, list_distance_for_show