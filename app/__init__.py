from flask import Flask
from app.routes.doc2vec.mdeberta import mdeberta_blueprint
from app.routes.doc2vec.wangchanberta import wangchanberta_blueprint
from app.routes.sentencetransformer.mdeberta import bertmdeberta_blueprint
from app.routes.sentencetransformer.wangchanberta import bertwangchanberta_blueprint

def create_app():
    flask_app = Flask(__name__)
    flask_app.register_blueprint(mdeberta_blueprint)
    flask_app.register_blueprint(wangchanberta_blueprint)
    flask_app.register_blueprint(bertmdeberta_blueprint)
    flask_app.register_blueprint(bertwangchanberta_blueprint)

    return flask_app