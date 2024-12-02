import os
import json
from datetime import datetime

class TrainingLogger:
    def __init__(self,
                 model_out_path:str, 
                 model_name:str,
                 overwrite:bool=False,
                 pretrained:str=None,
                 train_dataset:str=None, 
                 eval_dataset:str=None):
        
        logger_path = os.path.join(model_out_path, model_name + ".json")
        if not os.path.exists(model_out_path):
            os.mkdir(model_out_path)
        
        # Attributes
        self.model_out_path = model_out_path
        self.logger_path    = logger_path
        
        # Training Data
        self.data = {
            "metadata": {
                "start-timestamp": None,
                "finish-timestamp": None,
                "pretrained": pretrained,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,

                "hyperparams": None
            },
            "epochs": {}
        }

        # Create the file if doesnt exist. Else load it.
        if os.path.exists(self.logger_path):
            if not overwrite:
                with open(self.logger_path, 'r') as f:
                    self.data = json.load(f)
            else:
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

    def get_epoch_metrics(self):
        metrics = {
            "train": {},
            "eval": {}
        }
        
        assert "epochs" in self.data 
        assert "0" in self.data["epochs"]
        assert "metrics" in self.data["epochs"]["0"]
        assert "train" in self.data["epochs"]["0"]["metrics"]

        for m in self.data["epochs"]["0"]["metrics"]["train"]:
            metrics["train"][m] = []
        
        for m in self.data["epochs"]["0"]["metrics"]["eval"]:
            metrics["eval"][m] = []

        for k,v in self.data["epochs"].items():
            train_metrics   = v["metrics"]["train"]
            eval_metrics    = v["metrics"]["eval"]
            for m in metrics["train"]:
                metrics["train"][m].append( train_metrics[m] )
            for m in metrics["eval"]:
                metrics["eval"][m].append( eval_metrics[m] )

        return metrics