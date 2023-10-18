from flask import Flask, request, jsonify

import os, difflib, gensim, pythainlp, multiprocessing
import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize,word_detokenize
from gensim.models import Word2Vec
import tensorflow as tf
from keras.layers import Embedding, Input, LSTM, Dense, TimeDistributed
from keras.models import Model

from pythainlp.word_vector import WordVector
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import torch

def create_app():
    flaskApp = Flask(__name__)
    df = pd.read_json('app/QuestionAnswer-SubmitFormm-deberta.json')

    wv = WordVector()

    def prepare_sentence_vectors(df):
        context_vectors = [wv.sentence_vectorizer(str(c)) for c in df]
        return context_vectors
    context_vectors = prepare_sentence_vectors(df['context'])
    questions_vectors = prepare_sentence_vectors(df['question'])

    tokenizer = AutoTokenizer.from_pretrained("powerpuf-bot/m-deberta_dataxet_FAQ_chatbot_2",token='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')
    model = AutoModelForQuestionAnswering.from_pretrained("powerpuf-bot/m-deberta_dataxet_FAQ_chatbot_2",token='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')

    ques_vec_norm = np.vstack(questions_vectors).astype('float32')
    ques_vec_norm = normalize(ques_vec_norm)
    context_vectors_for_faiss = np.vstack(context_vectors).astype('float32')
    context_vectors_for_faiss = normalize(context_vectors_for_faiss)

    # res = faiss.StandardGpuResources()
    def predict(message):
        index = faiss.IndexFlatL2 (ques_vec_norm.shape[1])
        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
        # gpu_index_flat.add(ques_vec_norm)

        index.add(ques_vec_norm)

        question_vector = wv.sentence_vectorizer(message)
        question_vector = normalize(question_vector)
        question_vector = question_vector.reshape(1, -1).astype('float32')

        k = 1  # Number of similar contexts to retrieve
        distances, indices = index.search(question_vector, k)

        similar_question_index = indices[0][0]
        similar_question = df['question'][similar_question_index]
        similar_context = df['context'][similar_question_index]
        distance = distances[0][0]  # Distance value

        inputs = tokenizer(message, similar_context, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        Answer = tokenizer.decode(predict_answer_tokens)
        result = {"question": similar_question, "distance": distance, "answer": Answer}
        print(result)
        result =Answer
        return result


    
    @flaskApp.route('/api/get_response_mde', methods=['POST'])
    def get_response_mde():
        data = request.get_json()
        user_message = data['message']

        bot_response = predict(user_message)

        return jsonify({'response': bot_response})
    
    return flaskApp