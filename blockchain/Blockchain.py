import blockchain.Block as b
from blockchain.model import User
import blockchain.DbController as db
import json
import blockchain.aes as aes
import pickle
key = '7625e224dc0f0ec91ad28c1ee67b1eb96d1a5459533c5c950f44aae1e32f2da3'

with open('blockchain/AES.pkl', 'rb') as inp:
    aes_obj = pickle.load(inp)


class Blockchain:
    def __init__(self):
        print('Inside blockchain 15...............')
        self.diff = 20
        self.maxNonce = 2 ** 32
        self.target = 2 ** (256 - self.diff)
        print('Inside blockchain 19...............')

        self.chain = []
        print('Inside blockchain 22...............')
        self.load_chain_from_database()
        print('Inside blockchain 24...............')


    def load_chain_from_database(self):

        cloud_chain_list = db.pull_data_from_cloud()

        for index in range(len(cloud_chain_list)):
            admin_user = User("root",
                              "root@root",
                              "root",
                              "admin",
                              "0.jpg")

            root_user = User(aes_obj.decrypt(str(cloud_chain_list[index]["data"]["name"]).replace('"','').replace("b'","").replace("'", "")),
                             aes_obj.decrypt(str(cloud_chain_list[index]["data"]["email"]).replace('"','').replace("b'","").replace("'", "")),
                             aes_obj.decrypt(str(cloud_chain_list[index]["data"]["password"]).replace('"','').replace("b'","").replace("'", "")),
                             aes_obj.decrypt(str(cloud_chain_list[index]["data"]["user_level"]).replace('"','').replace("b'","").replace("'", "")),
                             aes_obj.decrypt(str(cloud_chain_list[index]["data"]["image"]).replace('"','').replace("b'","").replace("'", "")))

            if index == 0:
                self.block = b.Block(admin_user)
                self.chain.append(self.block)
                self.head = self.block
            else:
                self.mine(b.Block(root_user), push_status=False)

        print('loaded')

    def add(self, block, push_status):

        block.previous_hash = self.block.hash()
        block.blockNo = self.block.blockNo + 1

        self.block.next = block
        self.block = self.block.next

        self.chain[-1].next = block
        self.chain.append(block)
        if push_status:

            encrypt_name = aes_obj.encrypt(str(json.loads(block.get_str())["data"]["name"]))
            encrypt_email = aes_obj.encrypt(str(json.loads(block.get_str())["data"]["email"]))
            encrypt_password = aes_obj.encrypt(str(json.loads(block.get_str())["data"]["password"]))
            encrypt_user_level = aes_obj.encrypt(str(json.loads(block.get_str())["data"]["user_level"]))
            encrypt_image = aes_obj.encrypt(str(json.loads(block.get_str())["data"]["image"]))
            push_string = '{ "_id" : ' + str(json.loads(
                block.get_str())["_id"]) + ', "data" : { "name" : "' + str(encrypt_name) + '", "email" : "' + str(
                encrypt_email) + '", "password" : "' + str(encrypt_password) + '", "user_level" : "' + str(
                encrypt_user_level) + '", "image" : "' + str(encrypt_image) + '"} }'
            db.push_data_to_cloud(json.loads(push_string))

    def mine(self, block, push_status=True):
        for n in range(self.maxNonce):
            if int(block.hash(), 16) <= self.target:
                self.add(block, push_status)
                break
            else:
                block.nonce += 1

    def get_chain(self):
        return self.chain

    def get_last_block(self):
        return self.chain[-1]

    def login(self, email, password):
        login_status = False
        image_path = ''
        for block in self.chain:
            if email == str(json.loads(block.get_str())["data"]["email"]) and password == str(
                    json.loads(block.get_str())["data"]["password"]):
                login_status = True
                image_path = str(json.loads(block.get_str())["data"]["image"])

        return login_status, image_path

    def get_user_type_by_id(self, id):
        user_type = 'not_found'
        for block in self.chain:
            if str(id) == str(json.loads(block.get_str())["_id"]):
                user_type = str(json.loads(block.get_str())["data"]["user_level"])

        return user_type

# for testing
#obj = Blockchain()
# # add user
# user = User("rrr", "roshan@root.com", "root", "user", "root_img")
# new = b.Block(user)
# obj.mine(new)

# get users
# block_list = obj.get_chain()
# for i in block_list:
 #   print(i.get_str())

# login
# a, b = obj.login('root@root.com', 'root')
# print(a, b)
