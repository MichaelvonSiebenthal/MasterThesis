
# Bank Telemarketing dataset

This folder contains the files regarding the Bank Telemarketing dataset. Following
files are present in this folder:

1. Main_File.ipynb contains the primary Python script for evaluating the Bank 
    Telemarketing dataset.

2. Testing_Graph_Structures.ipynb is a script which was created for experimenting 
    with different graphs structures via the MAG model. The script is useful for 
    experimentation. As the results from this script are not directly used for the 
    master's thesis. The code is not annotated. The script is however straight 
    forward and rather short and should be self-explanatory.

3. The bank-full.csv file contains all 45211 observations of the Bank Telemarketing 
    dataset.

4. The bank-names.txt contains the description of the dataset including the variables.

5. bank.csv contains a sample of 4'521 observations from the bank-full.csv dataset and
    is the one primarily used by the authors Moro et al. (2011,2014). 

6. The embeddings file contains the node embeddings from the Node2vec algorithm. 
    For the Node2vec algorithm, the graph which is generated using the bank.csv file 
    is used.

7. The model.pt file contains the saved pytorch model parameters for the GraphSage model.

8. node2vec.ipynb contains the script for the Node2vec results.
