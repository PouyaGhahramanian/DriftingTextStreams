import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd
from drifter import Drifter
import numpy as np

def generate_and_save_stream(filename_in, filename_out, label_column, drift_type, drift_start, drift_end, drift_intensity, drift_func,
                             num_drift_points, drift_distribution, sigmoid_scale, is_multilabel, logging, labels_num):
    df = pd.read_csv(filename_in)
    data_size = df.shape[0]
    labels = df[label_column].values
    drifter_instances = [Drifter(data_size, labels, drift_type, drift_start, drift_end, drift_intensity,
                                 drift_func, num_drift_points, drift_distribution, sigmoid_scale, is_multilabel, logging)
                         for _ in range(labels_num)]
    
    labels_drifting = np.zeros((data_size, labels_num))
    for i in range(data_size):
        if i % 100 == 0:
            print(f"i: {i}")
        for j in range(labels_num):
            labels_drifting[i, j] = drifter_instances[j].get_label(labels[i])
    
    label_names = [f'label_{i}' for i in range(labels_num)]
    labels_drifting_df = pd.DataFrame(labels_drifting, columns=label_names)
    df = pd.concat([df, labels_drifting_df], axis=1)
    
    df.to_csv(filename_out, index=False)
    print(df.head())
    print(df.columns)

if __name__ == '__main__':
    filename_in = "../data_input/filtrelenmis_temizlenmis_derlem.csv.gz"
    filename_out = "../data_output/turkish_news_multilabel.csv"
    generate_and_save_stream(filename_in, filename_out, label_column='category', drift_type='abrupt', drift_start=10000,
                             drift_end=70000, drift_intensity=0.005, drift_func='linear', num_drift_points=10,
                             drift_distribution='even', sigmoid_scale=10., is_multilabel=False, logging=True,
                             labels_num=10)