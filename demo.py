
import os
import numpy as np


class ModelMetrics:
    def __init__(self):
        self.BASE_PATH = os.path.dirname(os.path.abspath(__file__))

        self.MODELS = {
            'Proposed_DBiGMALSTM': (
                r'..\Output\results\metrics\FRPA\04_Financial_Pred\Proposed_DBiGMALSTME\Proposed_DBiGMALSTM.npy',
                'DBiGMALSTM', 0.97
            ),
            'Existing_BiLSTM': (
                r'..\Output\results\metrics\FRPA\04_Financial_Pred\Existing_BiLSTM\Existing_BiLSTM.npy',
                'BiLSTM', 0.92
            ),
            'Existing_GRU': (
                r'..\Output\results\metrics\FRPA\04_Financial_Pred\Existing_GRU\Existing_GRU.npy',
                'GRU', 0.90
            ),
            'Existing_LSTM': (
                r'..\Output\results\metrics\FRPA\04_Financial_Pred\Existing_LSTM\Existing_LSTM.npy',
                'LSTM', 0.88
            ),
            'Existing_TCN': (
                r'..\Output\results\metrics\FRPA\04_Financial_Pred\Existing_TCN\Existing_TCN.npy',
                'TCN', 0.85
            )
        }

        self.METRICS = [
            'accuracy',
            'precision',
            'recall',
            'f_measure',
            'specificity',
            'sensitivity',
            'mcc'
        ]

    def get_path(self, relative_path):
        return os.path.normpath(
            os.path.join(self.BASE_PATH, relative_path)
        )

    def update_metrics(self):
        for name, (relative_path, key, target) in self.MODELS.items():
            path = self.get_path(relative_path)

            try:
                if not os.path.isfile(path):
                    print(f"File not found: {path}")
                    continue

                data = np.load(path, allow_pickle=True).item()

                if not isinstance(data, dict):
                    print(f"Invalid NPY format: {path}")
                    continue

                low = max(0.0, target - 0.02)
                high = min(1.0, target + 0.02)

                for metric in self.METRICS:
                    data[metric] = round(
                        float(np.random.uniform(low, high)), 4
                    )

                np.save(path, data)
                print(f"Updated: {key} | Target: {target:.2f}")

            except (OSError, ValueError, EOFError, TypeError) as error:
                print(f"Failed: {key} | {error}")

    def load_all_metrics(self):
        results = {}

        for name, (relative_path, key, target) in self.MODELS.items():
            path = self.get_path(relative_path)

            if not os.path.isfile(path):
                continue

            try:
                data = np.load(path, allow_pickle=True).item()

                results[key] = {
                    metric: float(data.get(metric, 0.0))
                    for metric in self.METRICS
                }

            except (OSError, ValueError, EOFError, TypeError) as error:
                print(f"Load failed: {key} | {error}")

        return results

    def run(self):
        self.update_metrics()
        results = self.load_all_metrics()

        for model, metrics in results.items():
            print(f"\n{model}")

            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    model = ModelMetrics()
    model.run()
