from app import create_app

app = create_app()

if name == 'main':
    app.run(debug=True)