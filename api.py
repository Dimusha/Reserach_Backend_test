# import libraries
from flask import Flask, request
from flask_cors import CORS
import json
import werkzeug
from blockchain.Blockchain import Blockchain
import blockchain.Block as b
from face_age_gender_rec.face_recognition import face_rec
import os
from face_age_gender_rec.age_gender_recognition import ager_rec
from blockchain.model import User
import cv2
from tensorflow.keras.models import Sequential, save_model, load_model
import numpy as np
from ocr import easy_ocr
from ocr import tesseract_ocr
from voice_rec import google_cloud_speech_rec_english as v_rec_eng
print('loading')
app = Flask(__name__)
print('loading flask')
CORS(app)
print('222222222222222')
blockchain_obj = Blockchain()
print('33333333333333333333')
model = load_model('ocr/cnn/model.h5')
print('loading the model')

receipt_detect_type = 'cnn'  # or ocr


@app.route('/login', methods=['GET', 'POST'])
def login():
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    face_list = os.listdir('blockchain/face_images')
    face_rec_status = False
    rec_id = 0
    for i in face_list:
        if face_rec.verifyFace(i, filename):
            face_rec_status = True
            rec_id = i.split(".")[0]

    user_type = blockchain_obj.get_user_type_by_id(rec_id)

    if face_rec_status:
        age, gender = ager_rec.get_gender_and_age('upload/' + filename)
        return_str = '[{"status" : 1, "age" : ' + str(age) + ', "user_level" : "' + user_type + '"}]'
    else:
        return_str = '[{"status" : 0, "age" : 0, "user_level" : "Null"}]'

    return json.loads(return_str)


@app.route('/manual_login', methods=['GET', 'POST'])
def manual_login():
    email = request.form['email']
    password = request.form['password']

    blockchain_login_status, image_path = blockchain_obj.login(email, password)

    if blockchain_login_status:
        age, gender = ager_rec.get_gender_and_age(image_path)
        return_str = '[{"status" : 1, "age" : ' + str(age) + '}]'
    else:
        return_str = '[{"status": 0, "age": 0}]'

    return json.loads(return_str)


@app.route('/register', methods=['GET', 'POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    user_level = request.form['user_level']
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    img = cv2.imread('upload/' + filename)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

    if len(faces_rect) > 0:
        save_file_name = 'blockchain/face_images/' + str(len(blockchain_obj.get_chain())) + '.jpg'
        cv2.imwrite(save_file_name, img)
        user = User(name, email, password, user_level, save_file_name)
        blockchain_obj.mine(b.Block(user))
        age, gender = ager_rec.get_gender_and_age('upload/' + filename)
        return_str = '[{ "status" : 1}]'
    else:
        return_str = '[{"status" : 0 }]'
    return json.loads(return_str)


@app.route('/receipt_reader', methods=['GET', 'POST'])
def receipt_reader():
    imagefile = request.files['image']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save('upload/' + filename)

    if receipt_detect_type == 'cnn':
        img = cv2.imread('upload/' + filename)
        # img = cv2.imread('Test/' + name)
        img = np.asarray(img)
        img = cv2.resize(img, (32, 32))
        img = preProcessing(img)
        img = img.reshape(1, 32, 32, 1)
        predictions = model.predict(img)
        probVal = np.amax(predictions)
        print(probVal)
        final_pred = np.argmax(predictions, axis=1)
        print('----------------------')
        a = np.amax(predictions) * 100
        x = "%.2f" % round(a, 2)
        med_name_dict = {0: "Amoxicillin", 1: "Astifen", 2: "Candazole", 3: "Cetrizine", 4: "Paracetamol"}
        return_str = '[{ "medicine name" : "' + str(med_name_dict[final_pred[0]]) + '" }]'
    else:
        text = easy_ocr.get_ocr(filename)
        # text = tesseract_ocr.get_ocr(filename) # using tesseract
        return_str = '[{ "medicine name" : "' + str(text) + '" }]'

    return json.loads(return_str)


@app.route('/voice_reader', methods=['GET', 'POST'])
def voice_reader():
    imagefile = request.files['voice']

    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived voice File name : " + imagefile.filename)
    imagefile.save('audios/' + filename)

    res = v_rec_eng.get_name_from_voice(filename)

    return_str = '[{ "medicine name" : "' + str(res) + '"}]'
    return json.loads(return_str)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


if __name__ == "__main__":
    app.run(host="192.168.1.106", port=5500, debug=True)
