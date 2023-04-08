import json
# from flask import Flask
# from flask import request
from inference import get_recommendations

# app = Flask(__name__)
# @app.route('/')
# def index():
#     name = request.args.get('name')
#     print(name)
#     responce = get_recommendations(646903)
#     return json.dumps(responce)
# app.run()

from flask import Flask, request
appFlask = Flask(__name__)
@appFlask.route('/index')

def access_param():
    id = request.args.get('id')
    responce = get_recommendations(id)
    return json.dumps(responce)
    # return '''<h1>The source value is: {}</h1>'''.format(source)

appFlask.run(debug=True, port=5000)

# http://127.0.0.1:5000/with_parameters?name=Se&age=20