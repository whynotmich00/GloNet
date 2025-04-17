from typing import Any, Dict

class Config:
    def __init__(self):
        pass

    def as_dict(self, config: Any) -> Dict[str, Any]:
        """Convert a config object into a dictionary format."""
        if not all(hasattr(config, attr) for attr in ["features", "N", "epochs", "batch_size", "seed", "track_metrics", "optimizer", "learning_rate"]):
            raise AttributeError("Config object is missing required attributes.")

        configuration = {
            "model": self._extract_model_settings(config),
            "training_setting": self._extract_training_settings(config),
            "track_metrics": config.track_metrics,
        }

        optimizer_settings = self._extract_optimizer_settings(config)
        if optimizer_settings:
            configuration["optimizer_setting"] = optimizer_settings

        return configuration

    def _extract_model_settings(self, config: Any) -> Dict[str, Any]:
        """Extract model-related settings."""
        settings = {
            "model": config.model,
            "features": config.features,
        }
        
        if config.model == "CNN" and hasattr(config, "kernel_size"):
            settings["kernel_size"] = config.kernel_size
            
        return settings

    def _extract_training_settings(self, config: Any) -> Dict[str, Any]:
        """Extract training-related settings."""
        return {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
        }

    def _extract_optimizer_settings(self, config: Any) -> Dict[str, Any]:
        """Extract optimizer-related settings, if applicable."""
        
        assert config.optimizer in ["SGD", "ADAM"], "Optimizer not in 'SGD' or 'ADAM'"

        settings = {
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
        }

        if config.optimizer == "SGD" and hasattr(config, "momentum"):
            settings["momentum"] = config.momentum

        return settings
