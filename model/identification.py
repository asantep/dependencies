import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import numpy as np


class LoadSamples:
    def __init__(self, path_to_data):
        self.samples = self._load_samples(path_to_data)

    def _load_samples(self, path_to_data):
        with open(path_to_data, 'rb') as file:
            dico = pickle.load(file)
        return dico

class LinearARModel(nn.Module):
    def __init__(self, lag):
        """
        Initialize the Linear AutoRegressive model with a specified lag.

        Parameters:
        - lag: The number of past time points to use for predicting the next time point.
        """
        super(LinearARModel, self).__init__()
        self.lag = lag
        self.linear = nn.Linear(lag, 1)

    def forward(self, x):
        """
        Forward pass through the Linear AR model.

        Parameters:
        - x: Input tensor of shape (batch_size, lag).

        Returns:
        - Output tensor of shape (batch_size, 1), the predicted next time point.
        """
        return self.linear(x)


class TimeSeriesForecaster:
    def __init__(self, time_series, lag=3,epochs=2500):
        """
        Initialize the TimeSeriesForecaster with a time series data, and lag.

        Parameters:
        - time_series: The time series data as a numpy array.
        - lag: The number of past time points to use as context for forecasting.
        """
        self.time_series = time_series
        self.lag = lag
        self.epochs = epochs
        self.ar_model = LinearARModel(lag)
        self._train_ar_model()

    def _train_ar_model(self, learning_rate=0.01):
        """
        Train the linear AR model on the available time series data.
        """
        X, y = [], []
        for i in range(len(self.time_series) - self.lag):
            X.append(self.time_series[i:i+self.lag])
            y.append(self.time_series[i+self.lag])
        X = torch.tensor(np.array(X), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.ar_model.parameters(), lr=learning_rate)

        X_batches, y_batches = torch.split(X,len(X)//10), torch.split(y,len(y)//10)
        # for epoch in range(self.epochs):
        #     optimizer.zero_grad()
        #     outputs = self.ar_model(X).squeeze()
        #     loss = criterion(outputs, y)
        #     loss.backward()
        #     optimizer.step()
        #
        #     if epoch % 100 == 0:
        #         print(f'Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}')
        for epoch in range(self.epochs):
            loss = 0
            optimizer.zero_grad()

            for a,b in zip(X_batches,y_batches):
                outputs = self.ar_model(a).squeeze()
                loss += criterion(outputs, b)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{self.epochs}], Loss: {loss.item():.4f}')

    def forecast_next_values_ar(self, steps=10):
        """
        Forecast the next 'steps' values using the linear autoregressive model.

        Parameters:
        - steps: The number of future time points to predict.

        Returns:
        - predictions: A list of predicted values.
        """
        predictions = []
        current_series = self.time_series.tolist()

        for _ in range(steps):
            # Prepare the context from the last 'lag' values
            context = torch.tensor(current_series[-self.lag:], dtype=torch.float32)
            with torch.no_grad():
                predicted_value = self.ar_model(context).item()
            predictions.append(predicted_value)
            current_series.append(predicted_value)  # Update the series with the predicted value

        return predictions

    def plot_time_series(self, ar_predictions=None, actual_values=None):
        """
        Plot the time series with the predicted values from AR models and the actual future values.

        Parameters:
        - ar_predictions: A list of predicted values by the AR model.
        - actual_values: The actual future values for comparison.
        """
        plt.plot(self.time_series, label="Original Time Series")
        prediction_indices = range(len(self.time_series), len(self.time_series) + len(actual_values))
        # plt.axvline(x=len(self.time_series), color='k')
        if actual_values is not None:
            plt.plot(prediction_indices, actual_values, color='green', label="Actual Future Values")
#             plt.scatter(prediction_indices, actual_values, color='green')
        if ar_predictions is not None:
            plt.plot(prediction_indices, ar_predictions, color='red', linestyle='dashed', label="AR Predicted Values")
#             plt.scatter(prediction_indices, ar_predictions, color='blue')

        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.title("Time Series with Predicted and \n Actual Values", fontsize=18)
        plt.legend()
        plt.grid(True, linestyle='--', color='purple', alpha=.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.savefig('/home/etienne/Images/LLM-VS-AR.pdf', dpi=125)
        plt.show()


def run(path_to_data, path_to_store, lag=3, visualization=False):

    # Loading sample data
    samples = LoadSamples(path_to_data).samples
    models = {}  # Dictionary to store trained models

    for h in tqdm(samples.keys(), desc="Processing each step", unit="step"):
        models[h] = {}
        for ser in tqdm(samples[h], desc=f"Processing time series", leave=False):
            # print(ser, h)
            time_series, actual_values = samples[h][ser]['train'], samples[h][ser]['test']

            # Initializing and training an AR model
            forecaster = TimeSeriesForecaster(time_series, lag=lag)

            # Forecast next values ahead using the trained AR models (same length as actual values)
            steps = len(actual_values)
            ar_predictions = forecaster.forecast_next_values_ar(steps=steps)

            if visualization:
                # Plot the time series with the predicted next values from both models and the actual values
                forecaster.plot_time_series(ar_predictions=ar_predictions,
                                            actual_values=actual_values)

            models[h][ser] = forecaster.ar_model  # Store the trained model in the dictionary
            print(h, ser, models[h][ser])
            break
        break


if __name__ == '__main__':
    # for f in os.listdir('/media/etienne/VERBATIM/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences/Results/Sampling/ETTh1'):
    path_to_data = '/media/etienne/VERBATIM/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences/Results/Sampling/ETTh1/440_20.pkl'
    path_to_store = '/media/etienne/VERBATIM/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences/Results/Models/ETTh1'
    run(
        path_to_data=path_to_data,
        path_to_store=path_to_store,
        lag=210,
        visualization=True
    )


