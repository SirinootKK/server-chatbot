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

class QADoc2VecModel:
    def __init__(self, model_path, tokenizer_path, doc2vec_model_path, dataset_path):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_doc2vec = Doc2Vec.load(doc2vec_model_path)
        self.df = pd.read_excel(dataset_path, sheet_name='mdeberta')
        _df = pd.read_excel(dataset_path, sheet_name='Default')
        self.df['answers'] = _df['Answer']
        
        self.questions_vectors = self._prepare_sentence_vectors(self.df['question'])
        self.ques_vec_norm = np.vstack(self.questions_vectors).astype('float32')
        self.ques_vec_norm = normalize(self.ques_vec_norm)
        
        self.context_vectors = self._prepare_sentence_vectors(self.df['context'])
        self.context_vectors_for_faiss = np.vstack(self.context_vectors).astype('float32')
        self.context_vectors_for_faiss = normalize(self.context_vectors_for_faiss)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df['question'], self.df['answers'], test_size=0.2, random_state=42
        )

    def _tokenize_doc2vec(self, string):
        unwanted_strings = {',' , ';'}
        tokenized_list = []
        tokenize_with_customdict = word_tokenize(string.upper())
        for token in tokenize_with_customdict:
            if not token.isspace():
                for s in unwanted_strings:
                    token = token.replace(s, '')
                tokenized_list.append(word_tokenize(token, engine="newmm"))
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
        Answer = self._predict_model(similar_question, similar_context)
        Answer = Answer.strip().replace("<unk>", "@")

        return Answer, list_context_for_show, distance, list_distance_for_show

    
# model =  AutoModelForQuestionAnswering.from_pretrained('app/model/medeberta')
# tokenizer =  AutoTokenizer.from_pretrained('app/model/medeberta')

# fname = "app/model/medeberta/dataxet_qa_doc2vec_model_100ep"
# model_doc2vec = Doc2Vec.load(fname)

# df = pd.read_excel('app/dataset.xlsx', sheet_name='mdeberta')
# _df = pd.read_excel('app/dataset.xlsx', sheet_name='Default')
# df['answers'] = _df['Answer']


# def tokenize_doc2vec(string):
#     unwanted_strings = {',' , ';'}
#     tokenized_list = []
   
#     tokenized_list = list(np.concatenate(tokenized_list))
#     tokenized_list = [elem for elem in tokenized_list if elem.strip()]
#     return tokenized_list

# def prepare_sentence_vectors(df):
#     context_vectors = []
#     for c in df:
#         model_doc2vec.random.seed(model_doc2vec.seed)
#         c_vector = model_doc2vec.infer_vector(tokenize_doc2vec(str(c))).reshape(1,-1)
#         context_vectors.append(c_vector)
#     return context_vectors

# questions_vectors = prepare_sentence_vectors(df['question'])
# ques_vec_norm = np.vstack(questions_vectors).astype('float32')
# ques_vec_norm = normalize(ques_vec_norm)
# context_vectors = prepare_sentence_vectors(df['context'])
# context_vectors_for_faiss = np.vstack(context_vectors).astype('float32')
# context_vectors_for_faiss = normalize(context_vectors_for_faiss)

# X_train, X_test, y_train, y_test = train_test_split(df['question'], df['answers'], test_size=0.2, random_state=42)
# X_test = list(X_test)
# y_test = list(y_test)

# def predict_model(question, similar_context):
#     inputs = tokenizer(question, similar_context, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     start_logits = outputs.start_logits
#     end_logits = outputs.end_logits

#     start_probs = F.softmax(start_logits, dim=1)
#     end_probs = F.softmax(end_logits, dim=1)

#     answer_start_index = start_probs.argmax()
#     answer_end_index = end_probs.argmax()

#     predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
#     Answer = tokenizer.decode(predict_answer_tokens)
    
#     return Answer

# def predict(message): 
#     list_context_for_show = []
#     list_distance_for_show = []
    
#     index = faiss.IndexFlatL2(ques_vec_norm.shape[1])
#     index.add(ques_vec_norm)
#     message = message.strip(" \t\n")

#     model_doc2vec.random.seed(model_doc2vec.seed)
#     question_vector = model_doc2vec.infer_vector(tokenize_doc2vec(message)).reshape(1, -1)
    
#     if np.isnan(question_vector).any() or np.isinf(question_vector).any():
#         return "Unable to process the input. Please try again.", [], "N/A", []

#     question_vector = normalize(question_vector)
#     question_vector = question_vector.reshape(1, -1).astype('float32')
#     k = 5 
#     distances, indices = index.search(question_vector, k)
#     for i in range(min(5, k)):
#         similar_context_for_show = indices[0][i]
#         similar_context = df['context'][similar_context_for_show]
#         list_context_for_show.append(similar_context)
#         list_distance_for_show.append(str(1 - distances[0][i]))

#     distance = str(1 - distances[0][0])
    
#     similar_question_index = indices[0][0]
#     similar_question = df['question'][similar_question_index]
#     similar_context = df['context'][similar_question_index]
#     Answer = predict_model(similar_question, similar_context)
#     Answer = Answer.strip().replace("<unk>", "@")

#     return Answer, list_context_for_show, distance, list_distance_for_show

