class UndoManager:
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
        self.current_index = -1

    def add_state(self, state):
        if self.current_index < len(self.history) - 1:
            self.history = self.history[:self.current_index + 1]
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.current_index = len(self.history) - 1

    def undo(self):
        if self.current_index > 0:
            self.current_index -= 1
            return self.history[self.current_index]
        return None

    def redo(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            return self.history[self.current_index]
        return None