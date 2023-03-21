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
python main.py -dataset dataset -model model
````
* **dataset**: including \[MGTAB, Twibot-20, Cresci-15\].  
* **model**: including \['GCN', 'GAT', 'SAGE', 'RGCN'\].  

e.g.  
````
python main.py -dataset MGTAB -model GCN
````
