from utils.utils import JsonEncoder
from models.pipeline import get_recommendations
from flask import Flask, request
import json

# init application
app = Flask(__name__)

# set url to get predictions
@app.route('/get_recommendation')
def run():
    user_id = int(request.args.get('id'))
    response = get_recommendations(user_id = user_id)
    print(response)
    return json.dumps(response, cls = JsonEncoder)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port = 8000)

# http://127.0.0.1:8000/get_recommendation?id=646321
