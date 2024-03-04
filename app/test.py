import random
import gradio as gr
import time
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from pythainlp import Tokenizer
import pickle
import re
from pythainlp.tokenize import sent_tokenize

DEFAULT_MODEL = 'wangchanberta-hyp'
DEFAULT_SENTENCE_EMBEDDING_MODEL = 'intfloat/multilingual-e5-base'

MODEL_DICT = {
    'wangchanberta': 'Chananchida/wangchanberta-xet_ref-params',
    'wangchanberta-hyp': 'Chananchida/wangchanberta-xet_hyp-params',
}

EMBEDDINGS_PATH = 'data/embeddings.pkl'
DATA_PATH='data/dataset.xlsx'


class ChatBot:
    def __init__(self):
        self.df = None
        self.model = None
        self.tokenizer = None
        self.embedding_model = None
        self.index = None
        self.k = 5

    def load_data(self, path=DATA_PATH):
        self.df = pd.read_excel(path, sheet_name='Default')
        self.df['Context'] = pd.read_excel(path, sheet_name='mdeberta')['Context']
        print(len(self.df))
        print('Load data done')

    def load_model(self, model_name=DEFAULT_MODEL):
        self.model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DICT[model_name])
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name])
        print('Load model done')

    def load_embedding_model(self, model_name=DEFAULT_SENTENCE_EMBEDDING_MODEL):
        self.embedding_model = SentenceTransformer(model_name)
        print('Load sentence embedding model done')

    def set_index(self, vector):
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(vector.shape[1])
            gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
            gpu_index_flat.add(vector)
            self.index = gpu_index_flat
        else:
            self.index = faiss.IndexFlatL2(vector.shape[1])
            self.index.add(vector)

    def get_embeddings(self, text_list):
        return self.embedding_model.encode(text_list)

    def prepare_sentences_vector(self, encoded_list):
        encoded_list = [i.reshape(1, -1) for i in encoded_list]
        encoded_list = np.vstack(encoded_list).astype('float32')
        encoded_list = normalize(encoded_list)
        return encoded_list

    def load_embeddings(self, file_path=EMBEDDINGS_PATH):
        with open(file_path, "rb") as fIn:
            stored_data = pickle.load(fIn)
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

    def predict(self, question):
        t = time.time()
        question = question.strip()
        question_vector = self.get_embeddings([question])
        question_vector = self.prepare_sentences_vector(question_vector)
        distances, indices = self.faiss_search(question_vector)

        Answer = self.model_pipeline(question, self.df['Context'][indices[0][0]])
        _time = time.time() - t
        output = {
            "user_question": question,
            "answer": Answer,
            "totaltime": round(_time, 3),
            "distance": round(distances[0][0], 4)
        }
        return Answer

    def predict_test(self, question):
        t = time.time()
        question = question.strip()
        question_vector = self.get_embeddings([question])
        question_vector = self.prepare_sentences_vector(question_vector)
        similar_questions, similar_contexts, distances, indices = self.faiss_search(self.index,question_vector)

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

        Answer = self.model_pipeline(question, mostSimSegment)

        if len(Answer) <= 2:
            Answer = mostSimSegment

        start_index = mostSimContext.find(Answer)
        end_index = start_index + len(Answer)

        _time = time.time() - t
        output = {
            "user_question": question,
            "answer": self.df['Answer'][indices[0][0]],
            "totaltime": round(_time, 3),
            "distance": round(distances[0][0], 4),
            "highlight_start": start_index,
            "highlight_end": end_index
        }
        return output

    def highlight_text(self, text, start_index, end_index):
        if start_index < 0:
            start_index = 0
        if end_index > len(text):
            end_index = len(text)
        highlighted_text = ""
        for i, char in enumerate(text):
            if i == start_index:
                highlighted_text += "<mark>"
            highlighted_text += char
            if i == end_index - 1:
                highlighted_text += "</mark>"
        return highlighted_text

    def chat_interface_before(self, question, history):
        response = self.predict(question)
        return response

    def chat_interface_after(self, question, history):
        response = self.predict_test(question)
        highlighted_answer = self.highlight_text(response["answer"], response["highlight_start"], response["highlight_end"])
        return highlighted_answer


if __name__ == "__main__":
    bot = ChatBot()
    bot.load_data()
    bot.load_model()
    bot.load_embedding_model()
    embeddings = bot.load_embeddings(EMBEDDINGS_PATH)
    bot.set_index(bot.prepare_sentences_vector(embeddings))

    examples = [
        'ขอเลขที่บัญชีของบริษัทหน่อย',
        'บริษัทตั้งอยู่ที่ถนนอะไร',
        'ขอช่องทางติดตามข่าวสารทาง Line หน่อย',
        'อยากทราบความถี่ในการดึงข้อมูลของ DXT360 ในแต่ละแพลตฟอร์ม',
        'อยากทราบความถี่ในการดึงข้อมูลของ DXT360 บน Twitter',
        # 'ช่องทางติดตามข่าวสารของเรา',
    ]

    demo_before = gr.ChatInterface(fn=bot.chat_interface_before, examples=examples)
    demo_after = gr.ChatInterface(fn=bot.chat_interface_after, examples=examples)

    interface = gr.TabbedInterface([demo_before, demo_after], ["Before", "After"])
    interface.launch()
