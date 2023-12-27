from flask import Blueprint, request, jsonify
from app.model.wangchanberta.wangchanberta_model import predict

wangchanberta_blueprint = Blueprint('wangchanberta', __name__)

@wangchanberta_blueprint.route('/api/get_response_wc', methods=['POST'])
def get_response_wc():
    data = request.get_json()
    user_message = data['message']

    bot_response, context, distance, distanceShow  = predict(user_message)
    print("wangchanberta")

    return jsonify({'wc_response': bot_response, 'wc_similar_context': context, 'wc_distance': distance, 'wc_ls_dis': distanceShow})
