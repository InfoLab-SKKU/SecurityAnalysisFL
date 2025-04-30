import flwr as fl
from flwr.server import ServerConfig

from ot import ServerReceiver
import pickle

class OTStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()

    def configure_fit(self, server_round, parameters, client_manager):
        print("\n[Server] Starting OT request to a client...")

        client = list(client_manager.clients.values())[0]
        receiver = ServerReceiver(choice_index=2, n=5)  # Want sample_2

        public_keys_serialized = [k.export_key() for k in receiver.public_keys]

        # Simulate a custom RPC to client
        result = client_proxy_request(client, public_keys_serialized)

        if result is not None:
            received_sample = receiver.decrypt_selected(result)
            print(f"[Server] Received sample via OT: {received_sample}")
        else:
            print("OT failed")

        return super().configure_fit(rnd, parameters, client_manager)

def client_proxy_request(client_proxy, public_keys_serialized):
    # Fake RPC: simulate request to client
    fn = client_proxy.get_properties  # Reuse something, or modify Flower
    try:
        return client_proxy.client.get_sample_via_ot(public_keys_serialized)
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    strategy = OTStrategy()
    config = ServerConfig(num_rounds=3)
    fl.server.start_server(server_address="localhost:8080", strategy=strategy, config=config)

if __name__ == "__main__":
    main()
