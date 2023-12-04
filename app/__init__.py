from flask import Flask, request, jsonify

#import os, difflib, gensim, pythainlp, multiprocessing
import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize,word_detokenize
from pythainlp.util import dict_trie
from gensim.models.doc2vec import Doc2Vec
from gensim.test.utils import get_tmpfile

from sklearn.model_selection import train_test_split
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import torch
import torch.nn.functional as F
# from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances


def create_app():
    flaskApp = Flask(__name__)
    
    model =  AutoModelForQuestionAnswering.from_pretrained('app\model\medeberta')
    tokenizer =  AutoTokenizer.from_pretrained('app\model\medeberta')

    # fname = get_tmpfile("app\model\dataxet_qa_doc2vec_model")
    fname = "app\model\medeberta\dataxet_qa_doc2vec_model"
    model_doc2vec = Doc2Vec.load(fname)

    df = pd.read_excel('app\dataset.xlsx', sheet_name='mdeberta')
    _df = pd.read_excel('app\dataset.xlsx', sheet_name='Default')
    df['answers'] = _df['Answer']
    custom_dict =  set(open('app\custom_dict.txt',encoding='utf-16').read().split())
    trie = dict_trie(dict_source=custom_dict)

    def tokenize_doc2vec(string):
        unwanted_strings = {',' , ';'}
        tokenized_list = []
        tokenize_with_customdict = word_tokenize(string.upper(), custom_dict=trie)
        for token in tokenize_with_customdict:
            if not token.isspace():
                for s in unwanted_strings:
                    token = token.replace(s, '')
                if token in custom_dict:
                    tokenized_list.append([token])
                else:
                    tokenized_list.append(word_tokenize(token, engine="newmm"))
        tokenized_list = list(np.concatenate(tokenized_list))
        tokenized_list = [elem for elem in tokenized_list if elem.strip()] # keep only non-whitespace elements
        return tokenized_list

    def prepare_sentence_vectors(df):
        context_vectors = []
        for c in df:
            model_doc2vec.random.seed(model_doc2vec.seed) # Reseed everytime you generating a new vector to guarantee a consistent vector.
            c_vector = model_doc2vec.infer_vector(tokenize_doc2vec(str(c))).reshape(1,-1)
            context_vectors.append(c_vector)
        return context_vectors
    
    questions_vectors = prepare_sentence_vectors(df['question'])
    ques_vec_norm = np.vstack(questions_vectors).astype('float32')
    ques_vec_norm = normalize(ques_vec_norm)
    context_vectors = prepare_sentence_vectors(df['context'])
    context_vectors_for_faiss = np.vstack(context_vectors).astype('float32')
    context_vectors_for_faiss = normalize(context_vectors_for_faiss)

    X_train, X_test, y_train, y_test = train_test_split(df['question'], df['answers'], test_size=0.2, random_state=42)
    X_test = list(X_test)
    y_test = list(y_test)

    def predict_model(question, similar_context):
        inputs = tokenizer(question, similar_context, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)

        answer_start_index = start_probs.argmax()
        answer_end_index = end_probs.argmax()
        
        start_prob = start_probs[0, answer_start_index].item()
        end_prob = end_probs[0, answer_end_index].item()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        Answer = tokenizer.decode(predict_answer_tokens)
        
        return Answer   
    
    def predict(message): 
        list_context_for_show = []
        index = faiss.IndexFlatL2 (ques_vec_norm.shape[1])
        index.add(ques_vec_norm)
        message = message.strip(" \t\n")

        model_doc2vec.random.seed(model_doc2vec.seed)
        question_vector = model_doc2vec.infer_vector(tokenize_doc2vec(message)).reshape(1,-1)
        
        question_vector = normalize(question_vector)
        question_vector = question_vector.reshape(1, -1).astype('float32')
        k = 5 
        distances, indices = index.search(question_vector, k)
        for i in range(min(3,k)):
            similar_context_for_show = indices[0][i]
            similar_context = df['context'][similar_context_for_show]
            list_context_for_show.append(similar_context)
        distance = str(1 - distances[0][0])
        similar_question_index = indices[0][0]
        similar_question = df['question'][similar_question_index]
        similar_context = df['context'][similar_question_index]
        Answer = predict_model(similar_question, similar_context)
        Answer = Answer.strip().replace("<unk>","@")
        # Answer = {"user_question":message,"similar_context": similar_context.strip(), "distance": distance, "answer": Answer.strip().replace("<unk>","@")}
        return Answer, list_context_for_show , distance


    @flaskApp.route('/api/get_response_mde', methods=['POST'])
    def get_response_mde():
        data = request.get_json()
        user_message = data['message']

        bot_response, context, distance= predict(user_message)
        # , distances_values
        print('context',context)
        # print('probability',probability)
        print('distance',distance)
        #'distance' : distances_values
        return jsonify({'response': bot_response ,'simitar_context': context , 'distance' : distance})
    
    @flaskApp.route('/api/get_response_wc', methods=['POST'])
    def get_response_wc():
        data = request.get_json()
        user_message = data['message']

        bot_response, context, distance= predict(user_message)
        # , distances_values
        print('get_response_wc',context)
        # print('probability',probability)
        print('get_response_wc',distance)
        #'distance' : distances_values
        return jsonify({'response': bot_response ,'simitar_context': context , 'distance' : distance})
    
    return flaskApp