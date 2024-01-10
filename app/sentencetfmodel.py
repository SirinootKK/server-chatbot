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
import evaluate

class Chatbot:
    def __init__(self, df_path=None, test_df_path=None, model_path=None, tokenizer_path=None, embedding_model_name=None, embeddingsPath=None,hf_token=None):
        self.df = None
        self.test_df = None
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.index = None
        self.k = 1  # top k most similar

        if all(arg is not None for arg in (df_path, test_df_path, model_path, tokenizer_path, embedding_model_name, embeddingsPath,hf_token)):
            self.set_df(df_path)
            self.set_test_df(test_df_path)
            self.set_model(model_path,hf_token)
            self.set_tokenizer(tokenizer_path,hf_token)
            self.set_embedding_model(embedding_model_name)
            self.set_index(self.prepare_sentences_vector(self.load_embeddings(embeddingsPath)))
            print('Initialize object done')

    def set_df(self, path):
        self.df = pd.read_excel(path, sheet_name='Default')
        self.df.rename(columns={'Response': 'Answer'}, inplace=True)
        self.df['context'] = pd.read_excel(path, sheet_name='mdeberta')['context']
        print('Load full data done')

    def set_test_df(self, path):
        self.test_df = pd.read_excel(path, sheet_name='Test')
        print('Load test data done')

    def set_model(self, model,hf_token):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model, token=hf_token)
        print('Load model done')

    def set_tokenizer(self, tokenizer,hf_token):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, token=hf_token)
        print('Load tokenizer done')

    def set_embedding_model(self, model):
        self.embedding_model = SentenceTransformer(model)
        print('Load sentence embedding model done')

            
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

    # Getters
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
        print('Prepare sentence vector done')
        return encoded_list

    def store_embeddings(self, embeddings):
        with open('embeddings.pkl', "wb") as fOut:
            pickle.dump({'sentences': self.df['Question'], 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        print('Store embeddings done')

    def load_embeddings(self, file_path):
        with open(file_path, "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
        print('Load (questions) embeddings done')
        return stored_embeddings

    def model_pipeline(self, question, similar_context):
        inputs = self.tokenizer(question, similar_context, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index: answer_end_index + 1]
        Answer = self.tokenizer.decode(predict_answer_tokens)
        print('Model predict done')
        return Answer

    def faiss_search(self, question_vector):
        distances, indices = self.index.search(question_vector, self.k)
        similar_questions = [self.df['Question'][indices[0][i]] for i in range(self.k)]
        similar_contexts = [self.df['context'][indices[0][i]] for i in range(self.k)]
        print('Faiss search similar vector done')
        return similar_questions, similar_contexts, distances, indices

    def predict_faiss(self, message):
        t = time.time()
        message = message.strip()
        question_vector = self.get_embeddings(message)
        question_vector = self.prepare_sentences_vector([question_vector])
        similar_questions, similar_contexts, distances, indices = self.faiss_search(question_vector)
        Answers = [self.df['Answer'][i] for i in indices[0]]
        _time = time.time() - t
        output = {
            "user_question": message,
            "distances": [distances[0][i] for i in range(self.k)],
            "similar_questions": [question.strip() for question in similar_questions],
            "answer": Answers[0],
            "totaltime": round(_time, 3)
        }
        print('Predict using just Faiss Done')
        return output

      # Function to predict using BERT embedding
    def predict_bert_embedding(self,message):
        t = time.time()
        message = message.strip()
        question_vector = self.get_embeddings(message)
        question_vector=self.prepare_sentences_vector([question_vector])
        similar_questions, similar_contexts, distances,indices = self.faiss_search(question_vector)

        Answer = self.model_pipeline(similar_questions, similar_contexts)

        _time = time.time() - t
        output = {
            "user_question": message,
            "distances": [distances[0][i] for i in range(self.k)],
            "similar_questions": [question.strip() for question in similar_questions],
            "answer": Answer,
            "totaltime": round(_time, 3)
        }
        print('Predict using Faiss and Model Done')
        return output

    def predict_semantic_search(self,message,corpus_embeddings):
        t = time.time()
        message = message.strip()
        query_embedding = self.embedding_model.encode(message, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=1)
        hit = hits[0][0]
        context=self.df['context'][hit['corpus_id']]
        score="{:.4f})".format(hit['score'])
        Answer = self.model_pipeline(message, context)
        _time = time.time() - t
        output = {
            "user_question": message,
            "answer": Answer,
            "score":  score,
            "totaltime": round(_time, 3)
        }
        print('Predict using predict_semantic_search and model Done')
        return output

    # Function for evaluation
    def eval(self,ref, output):
        exact_match_metric = evaluate.load("exact_match")
        bert_metric = evaluate.load("bertscore")
        rouge_metric = evaluate.load("rouge")
        tk = Tokenizer()
        tk.set_tokenize_engine(engine='newmm')

        result = {}
        exact_match = exact_match_metric.compute(
            references=ref['Answer'], predictions=output["answer"])
        result['exact_match'] = round(exact_match['exact_match'],4)

        bert_score = bert_metric.compute(
            predictions=output["answer"], references=ref['Answer'], lang="th", use_fast_tokenizer=True)
        df_bert_prec = pd.DataFrame(bert_score['precision'])
        df_bert_prec.fillna(method='ffill', inplace=True)
        bert_score['precision'] = df_bert_prec.values.tolist()
        for i in bert_score:
            if i == 'hashcode':
                break
            result[i] = round(np.mean(bert_score[i]),4)

        rouge_score = rouge_metric.compute(
            predictions=output["answer"], references=ref['Answer'], tokenizer=tk.word_tokenize)
        for i in rouge_score:
            result[i] = round(rouge_score[i],4)

        mean_time = round(np.mean(output['totaltime']), 3)
        result['mean_time'] = f'{mean_time*1000} ms.'
        print(result)
        return result