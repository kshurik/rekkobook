import json
from flask import Flask, request


app = Flask(__name__)

@app.route('/index')
def hello_world():
    responce = "<p> HELLO WORLD! </p>"
    return json.dumps(responce)

app.run(debug=True, host="0.0.0.0", port=5000)
