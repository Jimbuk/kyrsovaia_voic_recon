from flask import Flask, render_template, request
import requests

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")


@app.route('/data', methods=['POST'])
def handle_data():
    file = request.files['file']
    if file.filename.split(".")[-1] == "wav":
        files = {'file': file.stream.read()}
        response = requests.post("http://host.docker.internal:5000/html", files=files)
        return response.text
    return "<h1>we only support .wav files</h1>"


app.run('0.0.0.0', port=9090)
