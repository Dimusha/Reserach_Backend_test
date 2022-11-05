import datetime
import hashlib


class Block:
    blockNo = 0
    data = None
    next = None
    hash = None
    nonce = 0
    previous_hash = 0x0
    timestamp = datetime.datetime.now()

    def __init__(self, data):
        self.data = data

    def hash(self):
        h = hashlib.sha256()
        h.update(
            str(self.nonce).encode('utf-8') +
            str(self.data).encode('utf-8') +
            str(self.previous_hash).encode('utf-8') +
            str(self.timestamp).encode('utf-8') +
            str(self.blockNo).encode('utf-8')
        )
        # self.hash = h.hexdigest()
        return h.hexdigest()

    def __str__(self):
        return (str(self.hash()) + "_" + str(self.blockNo) + "_" + str(
            self.data.get_str()))

    def get_str(self):
        block_string = '{ "_id" : ' + str(self.blockNo) + ', "data" : ' + str(self.data.get_str()) + '}'

        return block_string
