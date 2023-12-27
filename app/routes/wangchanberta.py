from flask import Blueprint, request, jsonify
from app.doc2vecmodel import QADoc2VecModel

wangchanberta_blueprint = Blueprint('wangchanberta', __name__)
# Chananchida/wangchanberta_dataxet_FAQ_chatbot_v3
# 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
model_path = 'app\model\wangchanberta'
tokenizer_path = 'app\model\wangchanberta'
doc2vec_model_path = 'app\dataxet_qa_doc2vec_model_100ep'
dataset_path = 'app\dataset.xlsx'
qa_doc2vec_model = QADoc2VecModel(model_path, tokenizer_path, doc2vec_model_path, dataset_path)

@wangchanberta_blueprint.route('/api/get_response_wc', methods=['POST'])
def get_response_wc():
    data = request.get_json()
    user_message = data['message']

    bot_response, context, distance, distanceShow  = qa_doc2vec_model.predict(user_message)
    print("wangchanberta")

    return jsonify({'wc_response': bot_response, 'wc_similar_context': context, 'wc_distance': distance, 'wc_ls_dis': distanceShow})
