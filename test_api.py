# import libraries
from flask import Flask, request
from flask_cors import CORS
import json
import werkzeug

app = Flask(__name__)
CORS(app)


@app.route('/login', methods=['GET', 'POST'])
def login():
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    return_str = '{"status" : 1, "age" : 25, "user_level" : "user"}'

    return json.loads(return_str)


@app.route('/manual_login')
def manual_login():
    email = request.form['email']
    password = request.form['password']

    return_str = ' "status" : 1'

    return json.loads(return_str)


@app.route('/register')
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    user_level = request.form['user_level']
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    return_str = '{"status" : 1 }'
    return json.loads(return_str)


@app.route('/receipt_reader')
def receipt_reader():
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save(filename)

    return_str = '{ "medicine name" : "Amoxicillin" }'

    return json.loads(return_str)


@app.route('/voice_reader')
def voice_reader():
    imagefile = request.files['voice']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived voice File name : " + imagefile.filename)
    imagefile.save(filename)

    return_str = '{ "medicine name" : "Amoxicillin" }'
    return json.loads(return_str)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1100, debug=True)
