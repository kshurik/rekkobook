import json
from inference import get_recommendations
from flask import Flask, request


appFlask = Flask(__name__)

@appFlask.route('/index')

def access_param():
    id = request.args.get('id')
    responce = get_recommendations(id)
    return json.dumps(responce)

appFlask.run(debug=True, host="0.0.0.0", port=5000)

# http://127.0.0.1:5000/index?id=646321
# http://127.0.0.1:5000/index?id=1047345

