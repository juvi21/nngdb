from transformers import AutoModelForCausalLM, AutoTokenizer
from debugger.core import NNGDB
from console.command_handler import CommandHandler
from console.repl import NNGDBREPL
from experiment.config import load_config

def main():
    # Load configuration
    config = load_config('config.yaml')

    # Load TinyLlama model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    debugger = NNGDB(model)
    debugger.set_context('tokenizer', tokenizer)
    
    command_handler = CommandHandler(debugger)
    repl = NNGDBREPL(command_handler)
    
    print("Welcome to NNGDB (Neural Network GDB)")
    print("Type 'help' for a list of commands, or 'quit' to exit.")
    repl.run()

if __name__ == "__main__":
    main()