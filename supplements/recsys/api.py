from models.lfm import LFMModel
from models.ranker import Ranker

from utils.utils import JsonEncoder
from models.pipeline import get_recommendations
from flask import Flask, request
import json

# init application
app = Flask(__name__)

with app.app_context():
    lfm_model =  LFMModel()
    ranker = Ranker()

# set url to get predictions
@app.route('/get_recommendation')
def run():
    user_id = int(request.args.get('user_id'))
    top_k = int(request.args.get('top_k', 20))
    response = get_recommendations(
        user_id = user_id,
        lfm_model = lfm_model,
        ranker = ranker,
        top_k = top_k
    )
    return json.dumps(response, cls = JsonEncoder)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
