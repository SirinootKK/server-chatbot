from flask import Blueprint, request,jsonify
from app.semanticmodel import SemanticModel

semanticwangchanberta_blueprint = Blueprint('semanticwc',__name__)

model_path = 'app/model/wangchanberta'
embeddings_path = 'app/model/embeddings.pkl'
sentenceEmbeddingModel='intfloat/multilingual-e5-base'
dataset_path = 'app/dataset.xlsx'

semanticwc = SemanticModel(df_path=dataset_path, test_df_path=dataset_path, model_path=model_path, tokenizer_path=model_path, embedding_model_name=sentenceEmbeddingModel, embeddingsPath=embeddings_path)

@semanticwangchanberta_blueprint.route('/api/get_semantic_wc', methods=['POST'])
def get_semantic_mde():
    data = request.get_json()
    user_message = data['message']

    answer, context,score,distance = semanticwc.predict_bert_embedding(user_message)


    return jsonify({'semantic_wc': answer, 'score_wc':score ,'context_semantic_wc':context, 'info_distance_wc':distance})


