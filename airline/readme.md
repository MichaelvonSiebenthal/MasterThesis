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

4. main_script.ipynb is the main Python script used for analyzing the graphs and 
    generating the machine learning results.

5. Node2Vec.ipynb is the script used for the Node2Vec results. For this script, the 
    data files embeddings, embeddings_test, DataFrame, clean_data, Test_DF are required.
    All these datasets were generated using the airline_data.ipynb file. Note, that 
    for the Node2Vec model, these files are provided in this repository can be used for the analysis. The mentioned 
    files are also used for the GCN and GraphSage models in the main_script.ipynb. 
    Unfortunately, the matching adjacency matrices could not be added to the github 
    repositories due to file size constraint. For that reason, all the mentioned files 
    here must be first generated in the airline_data.ipynb script for use in the main_script.ipynb.
    To make things simpler and considering the time it takes to generate node embeddings, 
    the datasets are provided for re-creating the Node2Vec results. Of course, one can also 
    recreate all datasets with the airline_data.ipynb file. This will however take up some time. 

6. Simulation.ipynb is used for running the GraphSage robustness simulation.
