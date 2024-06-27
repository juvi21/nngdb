import copy

class ExperimentManager:
    def __init__(self, model):
        self.base_model = model
        self.experiments = {}
        self.current_experiment = None

    def create_experiment(self, name):
        if name in self.experiments:
            return f"Experiment '{name}' already exists."
        self.experiments[name] = copy.deepcopy(self.base_model.state_dict())
        return f"Experiment '{name}' created."

    def switch_experiment(self, name):
        if name not in self.experiments:
            return f"Experiment '{name}' does not exist."
        self.base_model.load_state_dict(self.experiments[name])
        self.current_experiment = name
        return f"Switched to experiment '{name}'."

    def list_experiments(self):
        return list(self.experiments.keys())

    def delete_experiment(self, name):
        if name not in self.experiments:
            return f"Experiment '{name}' does not exist."
        del self.experiments[name]
        if self.current_experiment == name:
            self.current_experiment = None
        return f"Experiment '{name}' deleted."

    def get_current_experiment(self):
        return self.current_experiment