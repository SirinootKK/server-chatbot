from flask import Flask, request, jsonify

import os, difflib, gensim, pythainlp, multiprocessing
import numpy as np
import pandas as pd
# from pythainlp.tokenize import word_tokenize,word_detokenize
# from gensim.models import Word2Vec
# import tensorflow as tf
# from keras.layers import Embedding, Input, LSTM, Dense, TimeDistributed
# from keras.models import Model

from pythainlp.word_vector import WordVector
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer,AutoModelForQuestionAnswering
import torch

def create_app():
    flaskApp = Flask(__name__)

    # dataset_path = 'app/dataset.xlsx'

    # raw = pd.read_excel(dataset_path)

    # raw.dropna(inplace=True)
    # train = raw[["Question", "Answer"]]
    # len_test = int(0.2 * len(train))
    # test_raw = raw[:len_test]
    # test_raw=test_raw[["Question","Answer"]]

    # questions = list()
    # answers = list()

    # for index, row in raw.iterrows():
    #     questions.append(row['Question'])
    #     answers.append(row['Answer'])

    # tokenized_questions = []
    # tokenized_answers = []
    # for q in questions:
    #     tokenizer_result = word_tokenize(q, engine='newmm-safe', keep_whitespace=False)
    #     tokenized_questions.append(' '.join(tokenizer_result))
    # for a in answers:
    #     tokenizer_result = word_tokenize(a, engine='newmm-safe', keep_whitespace=False)
    #     tokenized_answers.append(' '.join(tokenizer_result))
    # questions = tokenized_questions
    # answers = tokenized_answers

    # EMBEDDING_DIM = 20

    # inp = questions + answers
    # inp = [i.split() for i in inp]
    # inp.append(['<OOV>']) 
    # inp.append(['_']) 
    # inp.append(['<EOS>'])
    # outp1 = "corpus.th.model"

    # w2v_model = Word2Vec(inp, vector_size=EMBEDDING_DIM, window=10, min_count=1,workers=multiprocessing.cpu_count())
    # w2v_model.save(outp1)

    # vocab = w2v_model.wv.key_to_index.keys()

    # idx = 0
    # idx2word = {}
    # for v in vocab:
    #     idx2word[idx] = v
    #     idx = idx + 1


    # questions_train = []
    # answers_train = []
    # for index, row in train.iterrows():
    #     questions_train.append(row['Question'])
    #     answers_train.append(row['Answer'])

    # tokenized_questions_train = []
    # tokenized_answers_train = []
    # for q in questions_train:
    #     tokenizer_result = word_tokenize(q, engine='newmm-safe', keep_whitespace=False)
    #     tokenized_questions_train.append(' '.join(tokenizer_result))
    # for a in answers_train:
    #     tokenizer_result = word_tokenize(a, engine='newmm-safe', keep_whitespace=False)
    #     tokenized_answers_train.append(' '.join(tokenizer_result))

    # questions_train = tokenized_questions_train
    # answers_train = tokenized_answers_train

    # def padding_sequence(listsentence,maxseq):
    #     dataset = []
    #     for s in listsentence:
    #         n = maxseq - len(s.split())
    #         if n>0:
    #             dataset.append(s+" <EOS>"*n)
    #         elif n<0:
    #             dataset.append(s[0:maxseq])
    #         else:
    #             dataset.append(s)
    #     return dataset

    # X1, X2, Y = [], [], []
    # max_words = -1
    # for sentence in questions_train:
    #     sentence_len = len(sentence.split())
    #     if max_words < sentence_len:
    #         max_words = sentence_len 
    #     X1.append(sentence)
    # max_len_x1 = max_words

    # max_words = -1
    # for sentence in answers_train:
    #     sentence_len = len(sentence.split())
    #     if max_words < sentence_len:
    #         max_words = sentence_len 
    #     Y.append(sentence)
    # max_len_y = max_words

    # for sentence in answers_train:
    #     X2.append('_'+ ' ' + sentence[0:len(sentence)])

    # n_in = max_len_x1 + 2 
    # n_out = max_len_y + 2  

    # X1 = padding_sequence(X1, n_in)
    # X2 = padding_sequence(X2, n_out)
    # Y = padding_sequence(Y, n_out)

    # w2v_vocab = list(w2v_model.wv.key_to_index.keys())
    # num_words_in_w2v = len(w2v_vocab)

    # def word_index(listword):
    #     dataset = []
    #     for sentence in listword:
    #         tmp = []
    #         sentence = sentence.split()
    #         for w in sentence:
    #             tmp.append(word2idx(w))
    #         dataset.append(tmp)
    #     return np.array(dataset)

    # def word2idx(word):
    #     ind = 0
    #     try:
    #         ind = w2v_model.wv.key_to_index[word]
    #     except:
    #         try:
    #             sim = similar_word(word)
    #             ind = w2v_model.wv.key_to_index[sim]
    #         except:
    #             ind = w2v_model.wv.key_to_index["<OOV>"]
    #     return ind

    # def similar_word(word):
    #     sim_word = difflib.get_close_matches(word, w2v_vocab)
    #     try:
    #         return sim_word[0]
    #     except:
    #         return "<OOV>"

    # def embedding_model(w2v_model, EMBEDDING_DIM):
    #     vocab_list = [(k, w2v_model.wv[k]) for k, v in w2v_model.wv.key_to_index.items()]
    #     embeddings_matrix = np.zeros((len(w2v_model.wv.key_to_index.items()), w2v_model.vector_size))
    #     for i in range(len(vocab_list)):
    #         word = vocab_list[i][0]
    #         embeddings_matrix[i] = vocab_list[i][1]

    #     embedding_layer = Embedding(input_dim=len(embeddings_matrix),
    #                                 output_dim=EMBEDDING_DIM,
    #                                 weights=[embeddings_matrix],
    #                                 trainable=False, name="Embedding")
    #     return embedding_layer, len(embeddings_matrix)

    # def Encode_Decode_embedding_model(n_input, n_output, n_units, w2v_model, EMBEDDING_DIM):
    #     encoder_inputs = Input(shape=(None,), name="Encoder_input")

    #     encoder = LSTM(n_units, return_state=True, name='Encoder_lstm')
    #     Shared_Embedding, vocab_size = embedding_model(w2v_model, EMBEDDING_DIM)
    #     word_embedding_context = Shared_Embedding(encoder_inputs)
    #     encoder_outputs, state_h, state_c = encoder(word_embedding_context)
    #     encoder_states = [state_h, state_c]

    #     decoder_inputs = Input(shape=(None,), name="Decoder_input")
    #     decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name="Decoder_lstm")
    #     word_embedding_answer = Shared_Embedding(decoder_inputs)
    #     decoder_outputs, _, _ = decoder_lstm(word_embedding_answer, initial_state=encoder_states)
    #     decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax', name="Dense_layer"))
    #     decoder_outputs = decoder_dense(decoder_outputs)
    #     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #     encoder_model = Model(encoder_inputs, encoder_states)
    #     decoder_state_input_h = Input(shape=(n_units,), name="H_state_input")
    #     decoder_state_input_c = Input(shape=(n_units,), name="C_state_input")
    #     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    #     decoder_outputs, state_h, state_c = decoder_lstm(word_embedding_answer, initial_state=decoder_states_inputs)
    #     decoder_states = [state_h, state_c]
    #     decoder_outputs = decoder_dense(decoder_outputs)
    #     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    #     return model, encoder_model, decoder_model

    # def preparingword(listword):
    #     word =[]
    #     for w in listword:
    #         word.append(' '.join(word_tokenize(w, engine='newmm', keep_whitespace=False)))
    #     return word

    # def onehot_to_int(inputvector):
    #     return np.argmax(inputvector, axis=1)

    # def invert(inputlist):
    #     sentence = []
    #     for ind in inputlist:
    #         sentence.append(idx2word[ind])
    #     sen=word_detokenize(sentence).replace("<EOS>","")
    #     return (sen)

    # def predict_sequence(infenc, infdec, source, n_steps):
    #     state = infenc.predict(source, verbose=0)
    #     target_seq = np.array(word_index("_"))
    #     output = list()
    #     for t in range(n_steps):
    #         yhat, h, c = infdec.predict([target_seq] + state, verbose=0)
    #         output.append(yhat[0,0,:])
    #         state = [h, c]
    #         target_seq = np.array([[np.argmax(yhat[0,0,:])]])
    #     return np.array(output)

    # def predict(Question):
    #     input_data = preparingword([Question])
    #     input_data = padding_sequence(input_data, n_in)
    #     input_data = word_index(input_data)
    #     target = predict_sequence(infenc, infdec, input_data, n_out)
    #     int_target = onehot_to_int(target)
    #     ans = invert(int_target)
    #     return ans

    # train, infenc, infdec = Encode_Decode_embedding_model(n_in, n_out, 256, w2v_model, EMBEDDING_DIM)
    # script_directory = os.path.dirname(os.path.abspath(__file__))
    # weight_file_path = os.path.join(script_directory, "model", "model_enc_weight.h5")
    # infenc.load_weights(weight_file_path)
    # weight_file_path = os.path.join(script_directory, "model", "model_dec_weight.h5")
    # infdec.load_weights(weight_file_path)

    # @flaskApp.route('/api/get_response', methods=['POST'])
    # def get_response():
    #     data = request.get_json()
    #     user_message = data['message']

    #     bot_response = predict(user_message)

    #     return jsonify({'response': bot_response})

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