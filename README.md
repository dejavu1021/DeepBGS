Quick Start:
1. To run DeepID, you need a python (3.7 or higher) environment. 2.
2. Ensure that the extension packages including Numpy, Pandas and seaborn are installed for your current python environment.
3. Download the above programs to the runtime directory
4. Result.csv is the data used to train and prediction.
5. machine learning&mlp&dimension reduction file, model.py is used for four kinds of machine learning to predict the experimental data, as well as dimensionality reduction, selecting the band after the results of the evaluation, mlp.py is used for multi-layer perceptual machine for the experimental data for the classification of rice tares. Among them, pcavalue.csv,tsnevalue.csv and cars22csv,spa2.csv are the data after dimensionality reduction and band selection, respectively. Before the dimensionality reduction prediction, the training set is optimized by valid_dimension_reduction.py for dimensionality reduction parameters. comp.py is used to calculate the vegetation index features to save the corresponding feature data.
6. parameter_selection file is used for parameter optimization in this model.
7. deep_learning file is used to compare the deep learning network with the model, and pth file is the corresponding model parameter file after the training of the corresponding network model.
8. draw_picture&compute is used to calculate the evaluation metrics and drawings.


