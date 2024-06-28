import readline
import rlcompleter
from .command_handler import CommandHandler

class NNGDBREPL:
    def __init__(self, debugger):
        self.debugger = debugger
        self.command_handler = CommandHandler(debugger)
        self.command_history = []
        self.setup_readline()

    def setup_readline(self):
        readline.parse_and_bind("tab: complete")
        readline.set_completer(rlcompleter.Completer(self.__dict__).complete)

    def run(self):
        print("Welcome to NNGDB (Neural Network GDB)")
        print("Type 'help' for a list of commands, or 'quit' to exit.")
        
        while True:
            current_experiment = self.debugger.get_current_experiment()
            try:
                user_input = input(f"<nngdb:{current_experiment}> ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                self.command_history.append(user_input)
                command_parts = user_input.split()
                if command_parts:
                    command, args = command_parts[0], command_parts[1:]
                    result = self.command_handler.handle_command(command, args)
                    print(result)
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        print("Exiting NNGDB...")

    def get_command_completions(self, text, state):
        commands = self.command_handler.get_available_commands()
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        return matches[state] if state < len(matches) else None
    
    def get_command_history(self):
        return self.command_history
    
    def complete(self, text, state):
        options = [cmd for cmd in self.command_handler.get_available_commands() if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        else:
            return None
        
    def cmd_history(self, *args):
        """
        Show command history.
        Usage: history [n]
        """
        n = int(args[0]) if args else len(self.command_history)
        return "\n".join(self.command_history[-n:])