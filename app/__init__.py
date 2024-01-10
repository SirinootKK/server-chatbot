from flask import Flask
from app.routes.doc2vec.mdeberta import mdeberta_blueprint
from app.routes.doc2vec.wangchanberta import wangchanberta_blueprint
from app.routes.semantic_search.mdeberta import semanticmdeberta_blueprint
# from app.routes.semantic_search.wangchanberta import semanticwangchanberta_blueprint

def create_app():
    flask_app = Flask(__name__)
    flask_app.register_blueprint(mdeberta_blueprint)
    flask_app.register_blueprint(wangchanberta_blueprint)
    flask_app.register_blueprint(semanticmdeberta_blueprint)
    # flask_app.register_blueprint(semanticwangchanberta_blueprint)

    return flask_app