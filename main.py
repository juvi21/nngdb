import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import NNGDB
from cli import NNGDBREPL

def main():
    parser = argparse.ArgumentParser(description="Neural Network GDB (NNGDB)")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on")
    parser.add_argument("--server", type=str, help="Connect to a running server (host:port)")
    args = parser.parse_args()

    if args.server:
        host, port = args.server.split(':')
        debugger = NNGDB(None, args.model, args.device)
        debugger.connect(host, int(port))
        print(f"Connected to NNGDB server at {host}:{port}")
    else:
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        print(f"Initializing NNGDB")
        debugger = NNGDB(model, args.model, args.device)
        debugger.set_context('tokenizer', tokenizer)
        debugger.set_context('device', args.device)

    print("Starting NNGDB REPL")
    repl = NNGDBREPL(debugger)
    repl.run()

if __name__ == "__main__":
    main()