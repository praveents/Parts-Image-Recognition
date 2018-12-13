from flask import Flask
import base64
from flask import request
from flask import json
from flask import make_response
from predict import *
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

app = Flask(__name__)


@app.route('/')
def index():
    return "Object Identification Model"


@app.route('/objectidentification/api/predictObject', methods=['GET', 'POST'])
def predictObject():
    target_file = ''
    res = dict()
    param = json.loads(request.data)#.decode("utf-8")
    type = param['type']
    image = param['img']
    encoded_image = base64.b64decode(image + "===") #+ "==="
    if type == 'image/jpeg':
        target_file = r'input.jpeg'
    elif type == 'image/jpg':
        target_file = r'input.jpg'
    elif type == 'image/png':
        target_file = r'input.png'

    with open(target_file, "wb") as f:
        f.write(encoded_image)
    data = dict()
    data = predict_image(target_file)
    js = json.dumps(data)
    return js



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
