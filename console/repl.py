import readline
import rlcompleter
from .command_handler import CommandHandler

class NNGDBREPL:
    def __init__(self, command_handler: CommandHandler):
        self.command_handler = command_handler
        self.setup_readline()

    def setup_readline(self):
        readline.parse_and_bind("tab: complete")
        readline.set_completer(rlcompleter.Completer(self.__dict__).complete)

    def run(self):
        while True:
            try:
                user_input = input("(nngdb) ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                parts = user_input.split()
                if parts:
                    command, args = parts[0], parts[1:]
                    result = self.command_handler.handle_command(command, args)
                    print(result)
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        print("Exiting NNGDB...")