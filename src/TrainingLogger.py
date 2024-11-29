import os
import json
from datetime import datetime

class TrainingLogger:
    def __init__(self, 
                 model_out_path:str, 
                 model_name:str, 
                 overwrite:bool=False,
                 train_dataset:str=None, 
                 eval_dataset:str=None):
        
        logger_path = os.path.join(model_out_path, model_name + ".json")
        if not os.path.exists(model_out_path):
            os.mkdir(model_out_path)
        
        if not overwrite and os.path.exists(logger_path):
            raise Exception(f"[TrainingLogger] Log file {logger_path} already exists!!")

        # Attributes
        self.model_out_path = model_out_path
        self.logger_path    = logger_path
        
        # Training Data
        self.data = {
            "metadata": {
                "start-timestamp": None,
                "finish-timestamp": None,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "hyperparams": None
            },
            "epochs": {}
        }

        # Create the file if doesnt exist. Else overwrite it.
        if os.path.exists(self.logger_path):
            with open(self.logger_path, 'r') as f:
                # self.training_data = json.load(f)
                self._save()
        else:
            self._save()

    def _save(self):
        """Save data to JSON"""
        with open(self.logger_path, 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def set_hyperparams(self, hyperparams: dict):
        """ Set the hyperparams data"""
        self.data["metadata"]["hyperparams"] = hyperparams
        self._save()

    def start_training(self):
        """Start the training process"""
        self.data["metadata"]["start-timestamp"] = datetime.now().isoformat()
        self._save()

    def finish_training(self):
        """Finish the training process"""
        self.data["metadata"]["finish-timestamp"] = datetime.now().isoformat()
        self._save()

    def log_epoch(self, epoch, checkpoint_path, train_metrics, eval_metrics):
        """Dump the training epoch data"""
        self.data["epochs"][epoch] = {
            "checkpoint": checkpoint_path,
            "metrics": {
                "train": train_metrics,
                "eval": eval_metrics
            }
        }
        self._save()
