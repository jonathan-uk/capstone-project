# capstone-project
## MACHINE LEARNING ENGINEER NANODEGREE

The image classification problem was solved with PyTorch: I Load the dataset from fashion-MNIST website.The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
I Created a validation dataset from train data, creating separate folders for train, valid and test dataset Loading the datasets to s3 and performing job training on a preprocessed dataset with pytorch on AWS sagemaker Performing debugging and profiling Creating inference and prediction.
The dataset waas gotten from pytorch  https://pytorch.org/vision/stable/datasets.html#fashion-mnist

Included in this repository are:

The jupyter notebooks(data_images.ipynb and tune_train_deploy.ipynb) where run all my codes for this project

The python scripts used as entry_points for training and deploying endpoints(fm_hpo.py, fm_model.py and fm_inference.py) located in the python_scripts folder

The profiler report made after training the model

The code used in calculating the mean and standard deviation of the dataset(mean_and_std.ipynb)

The report and proposal of this project
