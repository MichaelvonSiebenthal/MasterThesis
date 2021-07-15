# Airline Dataset

This is the main dataset used for the master's thesis. This folder includes following
files.

1. train.csv and test.csv are the train and test datasets from the US Airline Passenger
    dataset as it is provided on KAGGLE.

2. description.txt contains the description of the dataset.

3. airline_data.ipynb is the Python script used to generate the graph and all the 
    data and is the starting point for the analysis. All datasets are generated with this
    script using the train.csv and test.csv datasets. For convenience, all the generated
    datasets from this script are provided separately in this repository. That way, all scripts 
    can be run individually and there are not interdependencies. Please note, that the adjacency
    matrices are currently compressed in a .zip file. These need to be first extracted for the 
    scripts to run properly. Each script will indicate at the beginning of the file which datasets 
    are required for the analysis.

4. The file DataFrame contains the raw subsample used as the training data.

5. The file TEST_DF is used for the Node2Vec.ipynb script and the Simulation.ipynb script. 
    This file contains the subsample of 6'000 nodes used as the test data.

6.  The file TEST_DF_det contains the test feature and is used for analyzing the 
    results of the deterministic MAG.

7. adjacency_matrices.zip contains the adjacency matrix 'adjacency_matrix.csv' which 
    is the adjacency matrix of the training graph. It further contains the adjacency matrix
    'adjacency_matrix_test_graph.csv' which is the adjacency matrix of the test graph. Lastly,
    the zip file contains the adjacency matrix of the deterministic graph as well as the
    adjacency matrix of the graph excluding the attribute gender. These .csv files must first 
    be unpacked before they can be used.

8. The file clean_data contains the cleaned training data generated from the airline_data.ipynb file.

9. embeddings contain the 2-dimensional node embeddings of the training graph which were generated using Node2Vec.

10. embeddings_test contain the 2-dimensional node embeddings of the test graph which were generated using Node2Vec.

11. embeddings_5 contains the 5-dimensional node embeddings of the training graph.

12. main_script.ipynb is the main Python script used for analyzing the graphs and 
    generating the machine learning results.

13. Node2Vec.ipynb is the script used for the Node2Vec results. 

14. Simulation.ipynb is used for running the GraphSage robustness simulation.
