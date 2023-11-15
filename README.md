# Model Capability Testing Suite

This repository provides a comprehensive suite of tools for evaluating machine learning models on regression and classification tasks. Included within are datasets, testing scripts, and plotting utilities designed to facilitate the assessment of model performance and capability.

## Features
- **Models**: change yourNN to import your model 
- **Datasets**: You could change every datasets for both regression and classification tasks. I have already give the example for calculate the performance matrices of regression and multi-classification tasks
- **Testing Scripts**: Automated scripts to test your models against provided or custom datasets.
- **Plotting Code**: Visualization scripts to plot the results for an intuitive understanding of model performance.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.9
- Relevant machine learning libraries as per the requirements.txt file
- Additional dependencies may be required depending on the dataset and model

### Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/guangjun1997327/model-capability-testing-suite.git
cd model-capability-test
pip install -r requirements.txt

### Usage
this code will help you to build your own test file with different performance matrices like rmse, nll, train time for the regression
and acc, train time for the classification. For the multi classification I provide the one-hot coding and de-one-hot coding algorithms.
:D so, you could get the prediction results directly with the number!
