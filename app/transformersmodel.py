import random
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
import pickle
import re
from pythainlp.tokenize import sent_tokenize, crfcut


class TransformersModel:
    SHEET_NAME_MDEBERTA = 'mdeberta'
    SHEET_NAME_DEFAULT = 'Default'
    UNKNOWN_ANSWERS = ["กรุณาลงรายระเอียดมากกว่านี้ได้มั้ยคะ", "ขอโทษค่ะลูกค้า ดิฉันไม่ทราบจริง ๆ"]

    def __init__(self, df_path=None, model_path=None, tokenizer_path=None, embedding_model_name=None, embeddingsPath=None):
        self.df = None
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.index = None
        self.k = 5
        if all(arg is not None for arg in (df_path, model_path, tokenizer_path, embedding_model_name, embeddingsPath)):
            self.set_df(df_path)
            self.set_model(model_path)
            self.set_tokenizer(tokenizer_path)
            self.set_embedding_model(embedding_model_name)
            sentences_vector = self.load_embeddings(embeddingsPath)
            repared_vector = self.prepare_sentences_vector(sentences_vector)
            self.set_index(repared_vector)
            
    def set_index(self, vector):
        if torch.cuda.is_available():  # Check if GPU is available
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(vector.shape[1])
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index_flat.add(vector)
            self.index = gpu_index_flat
        else:  # If GPU is not available, use CPU-based Faiss index
            self.index = faiss.IndexFlatL2(vector.shape[1])
            self.index.add(vector)
        return self.index 

    def set_df(self, path):
        self.df = pd.read_excel(path, sheet_name=self.SHEET_NAME_DEFAULT)
        self.df.rename(columns={'Response': 'Answer'}, inplace=True)
        self.df['Context'] = pd.read_excel(path, self.SHEET_NAME_MDEBERTA)['Context']

    def set_model(self, model):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def set_embedding_model(self, model):
        self.embedding_model = SentenceTransformer(model)

    def set_k(self, k_value):
        self.k = k_value

    def get_df(self):
        return self.df

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
        return Answer.replace('<unk>','@')

    def faiss_search(self, index, question_vector):
        if index is None:
            raise ValueError("Index has not been initialized.")
        distances, indices = index.search(question_vector, self.k)
        similar_questions = [self.df['Question'][indices[0][i]] for i in range(self.k)]
        similar_contexts = [self.df['Context'][indices[0][i]] for i in range(self.k)]
        return similar_questions, similar_contexts, distances, indices
    
    def faiss_segment_search(self, index, question_vector, x=1):
        if index is None:
            raise ValueError("Index has not been initialized.")
        distances, indices = index.search(question_vector, x)
        return distances, indices
    
    def create_segment_index(self, vector):
        segment_index = faiss.IndexFlatL2(vector.shape[1])
        segment_index.add(vector)
        return segment_index

    def predict_bert_embedding(self, question):
        list_context_for_show = []
        list_distance_for_show = []
        list_similar_question = []

        question = question.strip()
        question_vector = self.get_embeddings([question])
        question_vector = self.prepare_sentences_vector([question_vector])
        similar_questions, similar_contexts, distances, indices = self.faiss_search(self.index, question_vector)

        
        mostSimContext = similar_contexts[0]
        pattern = r'(?<=\s{10}).*'
        matches = re.search(pattern, mostSimContext, flags=re.DOTALL)
        if matches:
            mostSimContext = matches.group(0)
        mostSimContext = mostSimContext.strip()
        mostSimContext = re.sub(r'\s+', ' ', mostSimContext)
        
        segments = sent_tokenize(mostSimContext, engine="crfcut")

        segment_embeddings = self.get_embeddings(segments)
        segment_embeddings = self.prepare_sentences_vector(segment_embeddings)
        segment_index = self.create_segment_index(segment_embeddings)

        _distances, _indices = self.faiss_segment_search(segment_index, question_vector)
    
        mostSimSegment = segments[_indices[0][0]]

        print(f"_indices => {_indices[0][0]}")
        # answer = self.model_pipeline(question, self.df['Context'][indices[0][0]])
        answer = self.model_pipeline(question, mostSimSegment)
        
        if len(answer) <= 2:
            answer = mostSimSegment
        
        start_index = mostSimContext.find(answer)
        end_index = start_index + len(answer)

        # start_index = mostSimContext.find(mostSimSegment)
        # end_index = start_index + (len(mostSimSegment) - 1)
        print(f"mostSimContext {len(mostSimContext)} =>{mostSimContext}\nsegments {len(segments)} =>{segments}\nmostSimSegment {len(mostSimSegment)} =>{mostSimSegment}")
        print(f"answer {len(answer)} => {answer} || startIndex =>{start_index} || endIndex =>{end_index}")

        for i in range(min(5, self.k)):
            index = indices[0][i]
            similar_question = similar_questions[i]
            similar_context = similar_contexts[i]

            list_similar_question.append(similar_question)
            list_context_for_show.append(similar_context)
            list_distance_for_show.append(str(1 - distances[0][i]))

        distance = list_distance_for_show[0]

        if float(distance) < 0.5:
            answer = random.choice(self.UNKNOWN_ANSWERS)

        output = {
            "user_question": question,
            "answer": self.df['Answer'][indices[0][0]],
            "distance": distance,
            "highlight_start": start_index,
            "highlight_end": end_index,
            "list_context": list_context_for_show,
            "list_distance": list_distance_for_show
        }
        return output['answer'], output['list_context'], output['distance'], output['list_distance'], output['highlight_start'], output['highlight_end']
