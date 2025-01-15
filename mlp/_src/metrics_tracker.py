from statistics import mean

class MetricTracker:
    def __init__(self, name: str):
        """
        Generic tracker for metrics such as Loss and Accuracy.
        """
        self.metrics = {"Training": {}, "Validation": {}}
        self.name = name

    def add_entry(self, mode: str, epoch: int, step: int, value):
        """
        Add a metric entry for the given mode, epoch, and step.
        """
        assert mode in ("Training", "Validation"), "Mode must be either 'Training' or 'Validation'."
        
        epoch_key = f"Epoch{epoch}"
        step_key = f"Step{step}"
        
        # Initialize the epoch if it doesn't exist
        if epoch_key not in self.metrics[mode]:
            self.metrics[mode][epoch_key] = {}
        
        # Add the value for the specific step
        self.metrics[mode][epoch_key][step_key] = float(value)

    def mean_on_epochs(self):
        return {
            mode: {
                epoch: mean(self.metrics[mode][epoch].values()) for epoch in self.metrics[mode]
                } 
            for mode in self.metrics
            }

    def __repr__(self):
        return f"{self.name} Tracker: {self.metrics}"


class LossTracker(MetricTracker):
    def __init__(self):
        super().__init__("Loss")

    def __call__(self, mode: str, epoch: int, step: int, loss):
        self.add_entry(mode, epoch, step, loss)


class AccuracyTracker(MetricTracker):
    def __init__(self):
        super().__init__("Accuracy")

    def __call__(self, mode: str, epoch: int, step: int, accuracy):
        self.add_entry(mode, epoch, step, accuracy)