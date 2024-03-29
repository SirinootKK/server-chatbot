from flask import Blueprint, request,jsonify
from app.transformersmodel import TransformersModel

semanticmdeberta_blueprint = Blueprint('semanticmde',__name__)

model_path = 'app/model/medeberta'
# tokenizer_path = 'app/model/medeberta'
embeddings_path = 'app/model/embeddings.pkl'
sentenceEmbeddingModel='intfloat/multilingual-e5-base'
dataset_path = 'app/dataset.xlsx'

semanticmde = TransformersModel(df_path=dataset_path, model_path=model_path, tokenizer_path=model_path, embedding_model_name=sentenceEmbeddingModel, embeddingsPath=embeddings_path)

@semanticmdeberta_blueprint.route('/api/get_semantic_mde', methods=['POST'])
def get_semantic_mde():
    data = request.get_json()
    user_message = data['message']

    answer, context, score,distance,start_index,end_index= semanticmde.predict_bert_embedding(user_message)
    return jsonify({'semantic_mde': answer, 'score':score ,'context_semantic_mde':context, 'info_distance':distance, 'start_index':start_index, 'end_index':end_index})


