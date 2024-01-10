from flask import Blueprint, request, jsonify
from app.doc2vecmodel import QADoc2VecModel

mdeberta_blueprint = Blueprint('mdeberta', __name__)

model_path = 'app\model\medeberta'
doc2vec_model_path = 'app\dataxet_qa_doc2vec_model_100ep'
dataset_path = 'app\dataset.xlsx'
qa_doc2vec_model = QADoc2VecModel(model_path, model_path, doc2vec_model_path, dataset_path)

@mdeberta_blueprint.route('/api/get_response_mde', methods=['POST'])
def get_response_mde():
    data = request.get_json()
    user_message = data['message']


    bot_response, context, distance, distanceShow = qa_doc2vec_model.predict(user_message)
    print("mdeBerta")
    return jsonify({'response': bot_response, 'simitar_context': context, 'distance': distance, 'list_distance_for_show':distanceShow})
