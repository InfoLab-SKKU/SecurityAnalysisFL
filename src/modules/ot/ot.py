from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

class ClientSender:
    def __init__(self, data_list):
        self.data_list = data_list

    def oblivious_transfer_send(self, public_keys):
        encrypted_data = []
        for i, pub_key in enumerate(public_keys):
            cipher = PKCS1_OAEP.new(pub_key)
            enc = cipher.encrypt(self.data_list[i].encode())
            encrypted_data.append(enc)
        return encrypted_data

class ServerReceiver:
    def __init__(self, choice_index, n):
        self.choice_index = choice_index
        self.public_keys = []
        self.private_keys = []

        for i in range(n):
            if i == choice_index:
                key = RSA.generate(2048)
                self.private_key = key
                self.public_keys.append(key.publickey())
            else:
                dummy_key = RSA.generate(2048)
                self.public_keys.append(dummy_key.publickey())

    def decrypt_selected(self, encrypted_data):
        cipher = PKCS1_OAEP.new(self.private_key)
        decrypted = cipher.decrypt(encrypted_data[self.choice_index])
        return decrypted.decode()
