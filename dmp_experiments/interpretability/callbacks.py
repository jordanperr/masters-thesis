import interpretability.metrics as metrics
from interpretability.utils import NpEncoder
import tensorflow
import json
from collections import defaultdict

from pathlib import Path

class InterpretabilityMetricsKerasCallback(tensorflow.keras.callbacks.Callback):

    def __init__(self, epochs, data, metrics="all", save_to_path=False, save_to_history=True):
        self.epochs = epochs
        self.save_path = save_to_path
        self.data = data
        self.metrics = metrics
        self.save_to_history = save_to_history
        self.history = defaultdict(list)

    def compute_the_metrics(self, logs=None):

        X_train, X_test, y_train, y_test = self.data
        
        out = {}
        
        out["sparsity"] = metrics.sparsity(self.model)
        out["basic_statistics"] = metrics.basic_statistics(self.model)
        out["layerwise_pca"] = metrics.layerwise_pca(self.model, X_test, y_test)
        out["linear_probes"] = metrics.linear_probes(self.model, X_test, y_test)
        out["cca"] = metrics.cca(self.model, X_test, y_test)
        out["logs"] = logs
        
        return out

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.epochs == 0:

            out = {}
            if self.metrics == "all":
                out = self.compute_the_metrics()

            if self.save_path:
                base_path = self.save_path+f"/epoch={epoch}"
                Path(base_path).mkdir(parents=True, exist_ok=True)
                with open(base_path+'/logs.json', 'w') as outfile:
                    json.dump(out, outfile, cls=NpEncoder)
            
            if self.save_to_history:
                for key, value in out.items():
                    self.history[key].append(value)
                self.history["epoch"].append(epoch)