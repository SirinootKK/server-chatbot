from flask import Blueprint, request, jsonify
from app.model.wangchanberta.wangchanberta_model import predict

wangchanberta_blueprint = Blueprint('wangchanberta', __name__)

@wangchanberta_blueprint.route('/api/get_response_wc', methods=['POST'])
def get_response_mde():
    data = request.get_json()
    user_message = data['message']

    bot_response, context, distance = predict(user_message)
    print("wangchanberta",context)

    return jsonify({'response': bot_response, 'similar_context': context, 'distance': distance})