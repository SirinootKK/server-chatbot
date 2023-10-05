import os
from app import create_app

if __name__ == '__main__':
    flaskApp = create_app()

    flaskApp.run(debug=True)