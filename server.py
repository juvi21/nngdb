import socket
import threading
import pickle
import struct
from core.debugger import NNGDB
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class NNGDBServer:
    def __init__(self, model_name, device, port=5000):
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Initializing NNGDB")
        self.debugger = NNGDB(model, model_name, device)
        self.debugger.set_context('tokenizer', tokenizer)
        self.debugger.set_context('device', device)

        self.port = port
        self.clients = {}
        self.lock = threading.Lock()

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', self.port))
        server_socket.listen(5)
        print(f"Server started on port {self.port}")

        while True:
            client_socket, addr = server_socket.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
            client_thread.start()

    def handle_client(self, client_socket, addr):
        print(f"New connection from {addr}")
        while True:
            try:
                data = self.recv_msg(client_socket)
                if not data:
                    break
                command = pickle.loads(data)
                result = self.execute_command(command)
                self.send_msg(client_socket, pickle.dumps(result))
            except Exception as e:
                print(f"Error handling client {addr}: {e}")
                break
        print(f"Connection from {addr} closed")
        client_socket.close()

    def execute_command(self, command):
        with self.lock:
            method = getattr(self.debugger, command['method'])
            return method(*command['args'], **command['kwargs'])

    def send_msg(self, sock, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        sock.sendall(msg)

    def recv_msg(self, sock):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        return self.recvall(sock, msglen)

    def recvall(self, sock, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch NNGDB Server")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    server = NNGDBServer(args.model, args.device, args.port)
    server.start()