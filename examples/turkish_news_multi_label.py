"""
This script processes the Turkish news dataset containing labeled data, introduces concept drift using the Drifter class,
and saves the modified data to a new CSV file in the data_output folder. It supports various drift types (abrupt, gradual)
and functions (linear, sigmoid), with customizable parameters for drift intensity, start/end points, and distribution.
The resulting dataset can be used for evaluating machine learning models in dynamic environments with concept drift.
--------------------------------
Author: Pouya Ghahramanian
Date: 29 May 2024
"""
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import pandas as pd
import numpy as np
from drifter import Drifter

def generate_and_save_stream(filename_in, filename_out, label_column, drift_type, drift_start, drift_end, drift_intensity, drift_func,
                             num_drift_points, drift_distribution, sigmoid_scale, is_multilabel, logging, labels_num):
    df = pd.read_csv(filename_in)
    labels = df[label_column].values
    labels_unique = np.unique(labels)
    data_size = len(labels)
    drifter = None
    for j in range(labels_num):
        print(f"Processing drifter instance {j+1}")
        drifter = Drifter(data_size, labels_unique, drift_type, drift_start, drift_end, drift_intensity,
                                 drift_func, num_drift_points, drift_distribution, sigmoid_scale, is_multilabel, logging)
        labels_drifting = np.zeros((data_size))
        for i in range(data_size):
            if i % 100 == 0:
                print(f"Processing row {i}")
            labels_drifting[i] = drifter.get_label(labels[i])

        label_names = [f'label_{j+1}']
        labels_drifting_df = pd.DataFrame(labels_drifting, columns=label_names)
        df = pd.concat([df, labels_drifting_df], axis=1)
    
    df.to_csv(filename_out, index=False, compression='gzip')
    print(df.head())
    print(df.columns)

if __name__ == '__main__':
    filename_in = "../data_input/filtrelenmis_temizlenmis_derlem.csv.gz"
    # Abrupt Drift
    # filename_out = "../data_output/turkish_news_multilabel_abrupt.csv.gz"
    # generate_and_save_stream(filename_in, filename_out, label_column='category', drift_type='abrupt', drift_start=10000,
    #                          drift_end=70000, drift_intensity=0.005, drift_func='linear', num_drift_points=10,
    #                          drift_distribution='even', sigmoid_scale=10., is_multilabel=False, logging=True,
    #                          labels_num=10)
    # Gradual Linear Drift
    # filename_out = "../data_output/turkish_news_multilabel_gradual_linear.csv.gz"
    # generate_and_save_stream(filename_in, filename_out, label_column='category', drift_type='gradual', drift_start=10000,
    #                          drift_end=70000, drift_intensity=0.005, drift_func='linear', num_drift_points=10,
    #                          drift_distribution='even', sigmoid_scale=10., is_multilabel=False, logging=True,
    #                          labels_num=10)
    # Gradual Sigmoid Drift
    filename_out = "../data_output/turkish_news_multilabel_gradual_sigmoid.csv.gz"
    generate_and_save_stream(filename_in, filename_out, label_column='category', drift_type='gradual', drift_start=10000,
                             drift_end=70000, drift_intensity=0.005, drift_func='sigmoid', num_drift_points=10,
                             drift_distribution='even', sigmoid_scale=10., is_multilabel=False, logging=True,
                             labels_num=10)