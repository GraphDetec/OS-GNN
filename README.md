# OS-GNN
Over-Sampling Strategy in Feature Space for Graphs based Class-imbalanced Bot Detection

# Environment Settings 
* python == 3.7   
* torch == 1.8.1+cu102	  
* numpy == 1.21.6  
* scipy == 1.7.2  
* pandas == 1.3.5	
* scikit-learn == 1.0.2	 
* torch-cluster == 1.5.9	
* torch-geometric == 2.0.4	
* torch-scatter == 2.0.8	
* torch-sparse ==	0.6.12	
* torch-spline-conv	== 1.2.1	


# Usage 
````
python OS-GNN.py -dataset dataset -model model --smote smote
````
* **dataset**: including \[MGTAB, Twibot-20, Cresci-15\].  
* **model**: including \['GCN', 'GAT', 'SAGE', 'RGCN'\].  
* **smote**: including \[True, False\].  

e.g. 
Vanilla GCN on the MGTAB dataset
````
python OS-GNN.py -dataset MGTAB -model GCN -smote False
````
OS-GAT on the Twibot-20 dataset
````
python OS-GNN.py -dataset Twibot-20 -model GAT -smote True
````
