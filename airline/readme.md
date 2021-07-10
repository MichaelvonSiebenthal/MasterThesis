# Airline Dataset

This is the main dataset used for the master's thesis. This folder includes following
files.

1. train.csv and test.csv are the train and test datasets from the US Airline Passenger
    dataset as it is provided on KAGGLE.

2. description.txt contains the description of the dataset.

3. airline_data.ipynb is the Python script used to generate the graph and all the 
    data and is the starting point for the analysis. It is important, that this 
    dataset is run first to generate the required data for the subsequent Python 
    scripts. Due to the large file size of the generated adjacency matrices, these 
    could not be uploaded to GitHub due to file size limitations.

4. main_script.ipynb is the main Python script used for generating the results.

5. Node2Vec.ipynb is the script used for the Node2Vec results. 

6. Simulation.ipynb is used for running the GraphSage robustness simulations.
