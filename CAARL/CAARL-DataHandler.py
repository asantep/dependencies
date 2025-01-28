import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view
import pickle
from tqdm import tqdm

class DataHandler:
    def __init__(self, path_to_data, size, stride, normalize=True):
        self.size = size
        self.stride = stride
        self.data = self._load_data(path_to_data, normalize)

    def _load_data(self, path_to_data, normalize):
        data = pd.read_csv(path_to_data, index_col=0, parse_dates=True).fillna(0)

        if normalize:
            scaler = StandardScaler()
            data.iloc[:, :] = scaler.fit_transform(data)

        return data

    def get_time_intervals(self):
        # Use numpy array indexing for better performance
        return sliding_window_view(self.data.index, self.size)[::self.stride]

    def get_subsequences(self, time_interval):
        return self.data.loc[time_interval]


def process_size_stride(path_to_data, path_to_store, size, stride, normalize=True):
    try:
        data_handler = DataHandler(path_to_data, size, stride, normalize)

        intervals = data_handler.get_time_intervals()
        doc = {}
        step = 1

        for interval in intervals:
            train = data_handler.get_subsequences(interval).head(size - stride)
            test = data_handler.get_subsequences(interval).tail(stride)

            doc[step] = {h: {'train': train[h].to_numpy(), 'test': test[h].to_numpy()} for h in train.columns}

            step += 1

        # Save the doc dictionary to a pickle file
        filename = path_to_store+'/'+f'{size}_{stride}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(doc, f)
        # print(f"Saved {filename}")
        return filename

    except Exception as e:
        print(f"An error occurred in process_size_stride: {e}")
        return None


def main(path_to_data, path_to_store, stride = 20, sizes = list(range(100, 500, 20)), normalize=True):
    # Process each combination of size and stride sequentially
    for size in tqdm(sizes, desc="Processing sizes", unit="size"):
        filename = process_size_stride(path_to_data, path_to_store, size, stride, normalize)
        if filename is not None:
            print(f"Completed processing and saved\n: {filename}")
        else:
            print(f"Processing failed for size {size} and stride {stride}.")


if __name__ == '__main__':

    path_to_data = '/media/etienne/VERBATIM/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences/dataset/FXs_interpolated.csv'
    path_to_store = '/media/etienne/VERBATIM/Causal-Inference-Graph-Modeling-in-CoEvolving-Time-Sequences/Results/Sampling/Xchange-Rate'

    # Run the main function without parallel processing
    main(path_to_data, path_to_store, normalize=True)
