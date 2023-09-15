from flask import Flask, request, jsonify
import os, difflib, gensim, pythainlp, multiprocessing
import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize, word_detokenize
from gensim.models import Word2Vec
import tensorflow as tf
from keras.layers import Embedding, Input, LSTM, Dense, TimeDistributed
from keras.models import Model

app = Flask(__name__)

EMBEDDING_DIM = 20
outp1 = "corpus.th.model"

w2v_model = Word2Vec.load(outp1)

vocab = w2v_model.wv.key_to_index.keys()

idx = 0
idx2word = {}
for v in vocab:
    idx2word[idx] = v
    idx = idx + 1

def padding_sequence(listsentence, maxseq):
    dataset = []
    for s in listsentence:
        n = maxseq - len(s.split())
        if n > 0:
            dataset.append(s + " <EOS>" * n)
        elif n < 0:
            dataset.append(s[0:maxseq])
        else:
            dataset.append(s)
    return dataset

n_in = 256 
n_out = 256

w2v_vocab = list(w2v_model.wv.key_to_index.keys())
num_words_in_w2v = len(w2v_vocab)

def word_index(listword):
    dataset = []
    for sentence in listword:
        tmp = []
        sentence = sentence.split()
        for w in sentence:
            tmp.append(word2idx(w))
        dataset.append(tmp)
    return np.array(dataset)

def word2idx(word):
    ind = 0
    try:
        ind = w2v_model.wv.key_to_index[word]
    except:
        try:
            sim = similar_word(word)
            ind = w2v_model.wv.key_to_index[sim]
        except:
            ind = w2v_model.wv.key_to_index["<OOV>"]
    return ind

def similar_word(word):
    sim_word = difflib.get_close_matches(word, w2v_vocab)
    try:
        return sim_word[0]
    except:
        return "<OOV>"

def embedding_model(w2v_model, EMBEDDING_DIM):
    vocab_list = [(k, w2v_model.wv[k]) for k, v in w2v_model.wv.key_to_index.items()]
    embeddings_matrix = np.zeros((len(w2v_model.wv.key_to_index.items()), w2v_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        embeddings_matrix[i] = vocab_list[i][1]

    embedding_layer = Embedding(input_dim=len(embeddings_matrix),
                                output_dim=EMBEDDING_DIM,
                                weights=[embeddings_matrix],
                                trainable=False, name="Embedding")
    return embedding_layer, len(embeddings_matrix)

def Encode_Decode_embedding_model(n_input, n_output, n_units, w2v_model, EMBEDDING_DIM):
    encoder_inputs = Input(shape=(None,), name="Encoder_input")

    encoder = LSTM(n_units, return_state=True, name='Encoder_lstm')
    Shared_Embedding, vocab_size = embedding_model(w2v_model, EMBEDDING_DIM)
    word_embedding_context = Shared_Embedding(encoder_inputs)
    encoder_outputs, state_h, state_c = encoder(word_embedding_context)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None,), name="Decoder_input")
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name="Decoder_lstm")
    word_embedding_answer = Shared_Embedding(decoder_inputs)
    decoder_outputs, _, _ = decoder_lstm(word_embedding_answer, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax', name="Dense_layer"))
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(n_units,), name="H_state_input")
    decoder_state_input_c = Input(shape=(n_units,), name="C_state_input")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(word_embedding_answer, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return model, encoder_model, decoder_model

def preparingword(listword):
    word =[]
    for w in listword:
        word.append(' '.join(word_tokenize(w, engine='newmm', keep_whitespace=False)))
    return word

def onehot_to_int(inputvector):
    return np.argmax(inputvector, axis=1)

def invert(inputlist):
    sentence = []
    for ind in inputlist:
        sentence.append(idx2word[ind])
    sen = word_detokenize(sentence).replace("<EOS>", "")
    return (sen)

def predict_sequence(infenc, infdec, source, n_steps):
    state = infenc.predict(source, verbose=0)
    target_seq = np.array(word_index("_"))
    output = list()
    for t in range(n_steps):
        yhat, h, c = infdec.predict([target_seq] + state, verbose=0)
        output.append(yhat[0, 0, :])
        state = [h, c]
        target_seq = np.array([[np.argmax(yhat[0, 0, :])]])
    return np.array(output)

def predict(Question):
    input_data = preparingword([Question])
    input_data = padding_sequence(input_data, n_in)
    input_data = word_index(input_data)
    target = predict_sequence(infenc, infdec, input_data, n_out)
    int_target = onehot_to_int(target)
    ans = invert(int_target)
    return ans

train, infenc, infdec = Encode_Decode_embedding_model(n_in, n_out, 256, w2v_model, EMBEDDING_DIM)
script_directory = os.path.dirname(os.path.abspath(__file__))
weight_file_path = os.path.join(script_directory, "model", "model_enc_weight.h5")
infenc.load_weights(weight_file_path)
weight_file_path = os.path.join(script_directory, "model", "model_dec_weight.h5")
infdec.load_weights(weight_file_path)

@app.route('/api/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    user_message = data['message']

    bot_response = predict(user_message)

    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
