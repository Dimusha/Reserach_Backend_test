import pymongo
from pymongo import MongoClient
import json
import pickle
import blockchain.aes as aes

key = '7625e224dc0f0ec91ad28c1ee67b1eb96d1a5459533c5c950f44aae1e32f2da3'

# aes_obj = aes.AESCipher(key)
# with open('AES.pkl', 'wb') as outp:
#     aes_obj = aes.AESCipher(key)
#     pickle.dump(aes_obj, outp, pickle.HIGHEST_PROTOCOL)

with open('blockchain/AES.pkl', 'rb') as inp:
    aes_obj = pickle.load(inp)

# username - dimusha
# password - MuDaylTfkySaTFgM

cluster = MongoClient(
    "mongodb+srv://dimusha:MuDaylTfkySaTFgM@cluster0.jfr7wrz.mongodb.net/?retryWrites=true&w=majority")
db = cluster["chain"]
collection = db["users"]


def push_data_to_cloud(data):
    collection.insert_one(data)


def pull_data_from_cloud():
    res_list = []
    results = collection.find({})

    for i in results:
        res_list.append(i)
    return res_list


encrypt_name = aes_obj.encrypt("kaml")
encrypt_email = aes_obj.encrypt("kamal@gmail.com")
encrypt_password = aes_obj.encrypt("kam")
encrypt_user_level = aes_obj.encrypt("123")
encrypt_image = aes_obj.encrypt("1.jpg")

# x = '{ "_id" : 1, "data" : { "name" : "' + str(encrypt_name) + '", "email" : "' + str(
#     encrypt_email) + '", "password" : "' + str(encrypt_password) + '", "user_level" : "' + str(
#     encrypt_user_level) + '", "image" : "' + str(encrypt_image) + '"} }'
# print(x)
# push_data_to_cloud(json.loads(x))
# print(pull_data_from_cloud())
