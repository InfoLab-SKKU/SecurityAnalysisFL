import flwr as fl
from ot import ClientSender
import numpy as np

# Simulated local dataset (can be real too)
local_data = [f"sample_{i}" for i in range(5)]  # 5 data items

class OTFlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.data = local_data

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        return [], len(self.data), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.data), {}

    def get_sample_via_ot(self, public_keys_serialized):
        # Deserialize public keys
        from Crypto.PublicKey import RSA
        import pickle
        public_keys = [RSA.import_key(k) for k in public_keys_serialized]

        sender = ClientSender(self.data)
        encrypted_data = sender.oblivious_transfer_send(public_keys)

        # Return encrypted samples
        return encrypted_data

def main():
    fl.client.start_numpy_client(server_address="localhost:8080", client=OTFlowerClient())

if __name__ == "__main__":
    main()
