import os

from flask import Flask, send_from_directory

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Testing Prototype</h1>"

@app.route('/favicon.ico')
def fav():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico')

app.run()
