from flask import Flask
from app.routes.mdeberta import mdeberta_blueprint
from app.routes.wangchanberta import wangchanberta_blueprint

def create_app():
    flask_app = Flask(__name__)
    flask_app.register_blueprint(mdeberta_blueprint)
    flask_app.register_blueprint(wangchanberta_blueprint)

    return flask_app