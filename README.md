# DriftingTextStreams

DriftingTextStreams is a project dedicated to developing tools for creating and visualizing evolving text streams with induced concept drift in text classification datasets. This project, supported by Tübitak under grant 122E271, is currently in progress and aims to provide valuable resources for researchers and practitioners in the field.

## Citation
If you use our work in your research, please cite it as follows:

_BibTeX entry_

@article{10.1145/3639054,
author = {Ghahramanian, Pouya and Bakhshi, Sepehr and Bonab, Hamed and Can, Fazli},
title = {A Novel Neural Ensemble Architecture for On-the-fly Classification of Evolving Text Streams},
year = {2024},
issue_date = {May 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {18},
number = {4},
issn = {1556-4681},
url = {https://doi.org/10.1145/3639054},
doi = {10.1145/3639054},
journal = {ACM Trans. Knowl. Discov. Data},
month = feb,
articleno = {101},
numpages = {24},
keywords = {Data stream mining, concept drift, text stream classification, ensemble methods, neural networks}
}

_Formatted Citation_

Pouya Ghahramanian, Sepehr Bakhshi, Hamed Bonab, and Fazli Can. 2024. A Novel Neural Ensemble Architecture for On-the-fly Classification of Evolving Text Streams. ACM Trans. Knowl. Discov. Data 18, 4, Article 101 (May 2024), 24 pages. https://doi.org/10.1145/3639054


## Overview

DriftingTextStreams focuses on providing scripts that enable the creation of text streams with concept drift from existing text classification datasets and tools for visualizing the distribution of concept drift in these streams.

## Content

- **Drifter class**: To create data streams with concept drift based on a given real-world dataset.
- **Six Drifting Text Streams based on the Movies Dataset**: Our six generated text streams with distinct drift characteristics are available on [Google Drive](https://drive.google.com/drive/folders/1_Xcnb19WMLIhxOfPGuiE5CXg20cl9sqe?usp=sharing). 
- **Documentation**: Guidelines on how to use the scripts and interpret the results.

## Getting Started

To use DriftingTextStreams:

1. Clone the repository: `git clone https://github.com/pouyaghahramanian/DriftingTextStreams.git`
2. Navigate to the repository directory: `cd DriftingTextStreams`
3. Install required dependencies: `pip install -r requirements.txt`

## Usage

- **Creating Text Streams**: Use the `Drifter.py` class to create drifting data streams from any real-world dataset with different drift characteristics.
For instance, the following code segment generates a data stream with abrupt drifts, using an instance of the Drifter class:
```python
drifter = Drifter(
   total_data_size=42306,  # Total data size
   labels=['Comedy', 'Drama', 'Action', ...],  # List of labels
   drift_type='abrupt',  # Type of drift: 'abrupt' or 'gradual'
   drift_start=10000,  # Start timestep of drift
   drift_end=30000,  # End timestep of drift
   num_drift_points=10,  # Number of concept drift points
   drift_distribution='even'  # Distribution policy of the drift points
)
```
To get the drifting labels for the data instances, use the get_label() method:
```python
for i in range(drifter.total_data_size):
    drifting_label = drifter.get_label(original_label)
```
- **Visualizing Concept Drift**: Utilize the get_probability_log() method to analyze the concept drift and probability distributions in the generated text stream.

## Contributing

Your contributions are welcome! If you have suggestions, bug reports, or contributions, please feel free to open an issue or pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is supported by Tübitak under grant 122E271.
