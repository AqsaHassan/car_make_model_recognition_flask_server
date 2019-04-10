# Copyright © 2019 by Spectrico
# Licensed under the MIT License

from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
import detector
import base64
import traceback
import json
import io
import cv2

app = Flask(__name__)
CORS(app)

detector = detector.YOLO()

@app.route("/", methods = ['POST'])
@cross_origin()
def objectDetect():
    if request.headers['Content-Type'].startswith('multipart/form-data'):
        try:
            image_path = request.files['image']
            in_memory_file = io.BytesIO()
            image_path.save(in_memory_file)
            import numpy as np
            data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
            color_image_flag = 1
            img = cv2.imdecode(data, color_image_flag)
            if img is None:
                response = app.response_class(
                    response='415 Bad image',
                    status=415,
                    mimetype='text/plain'
                )
                return response
            result = detector.detect(img)
        except:
            traceback.print_exc()
            response = app.response_class(
                response='415 Unsupported Media Type',
                status=415,
                mimetype='text/plain'
            )
            return response
        response = app.response_class(
            response=json.dumps({'cars': result}),
            status=200,
            mimetype='application/json'
        )
        return response
    else:
        return "415 Unsupported Media Type"

@app.route("/", methods = ['GET'])
@cross_origin()
def version():
    response = app.response_class(
        response='{"version":"car make and model recognition 1.0"}',
        status=200,
        mimetype='application/json'
    )
    return response

@app.before_request
def option_autoreply():
    """ Always reply 200 on OPTIONS request """
    if request.method == 'OPTIONS':
        resp = app.make_default_options_response()

        headers = None
        if 'ACCESS_CONTROL_REQUEST_HEADERS' in request.headers:
            headers = request.headers['ACCESS_CONTROL_REQUEST_HEADERS']

        h = resp.headers

        # Allow the origin which made the XHR
        h['Access-Control-Allow-Origin'] = request.headers['Origin']
        # Allow the actual method
        h['Access-Control-Allow-Methods'] = request.headers['Access-Control-Request-Method']
        # Allow for 10 seconds
        h['Access-Control-Max-Age'] = "10"

        # We also keep current headers
        if headers is not None:
            h['Access-Control-Allow-Headers'] = headers

        return resp


@app.after_request
def set_allow_origin(resp):
    """ Set origin for GET, POST, PUT, DELETE requests """

    h = resp.headers

    # Allow crossdomain for other HTTP Verbs
    if request.method != 'OPTIONS' and 'Origin' in request.headers:
        h['Access-Control-Allow-Origin'] = request.headers['Origin']

    return resp

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000, debug=False, use_reloader=False, threaded=False)
