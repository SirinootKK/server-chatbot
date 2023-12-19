from flask import Blueprint, request, jsonify
from app.model.medeberta.mdeberta_model import predict

mdeberta_blueprint = Blueprint('mdeberta', __name__)

@mdeberta_blueprint.route('/api/get_response_mde', methods=['POST'])
# def get_response_mde():
#     data = request.get_json()
#     user_message = data['message']
#     bot_response, context, distance = predict(user_message)
#     print('mdeberta!')

#     return jsonify({'response': bot_response, 'similar_context': context, 'distance': distance})
def get_response_mde():
    data = request.get_json()
    print("Data received in get_response_mde:", data)
    user_message = data['message']

    bot_response, context, distance, distanceShow = predict(user_message)
    print('context', context)
    print('distance', distance)
    print('$show',distanceShow)

    return jsonify({'response': bot_response, 'simitar_context': context, 'distance': distance, 'list_distance_for_show':distanceShow})
