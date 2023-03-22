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

Run GNN and OS-GNN on bot detection datasets:
````
python OS-GNN.py -dataset dataset -model model --smote smote
````
* **dataset**: including \[MGTAB, Twibot-20, Cresci-15\].  
* **model**: including \['GCN', 'GAT', 'SAGE', 'RGCN'\].  
* **smote**: including \[True, False\].  


Vanilla GCN on the MGTAB dataset
````
python OS-GNN.py -dataset MGTAB -model GCN -smote False
````
OS-GAT on the Twibot-20 dataset
````
python OS-GNN.py -dataset Twibot-20 -model GAT -smote True
````



# Results
GCN
| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 68.76 </br> $_0.60$ |  68.30 </br> $_0.51$ | 68.29 </br> $_0.62$  |
| Cresci-15  | 96.50 </br> $_0.36$ |  96.20 </br> $_0.42$ | 95.95 </br> $_0.53$  |
| MGTAB      | 82.69 </br> $_0.76$ |  74.85 </br> $_1.32$ | 72.32 </br> $_1.29$  |
     

OS-GNN (backbone GCN)
| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 83.44 </br> $_0.40$ | 83.18 </br> $_0.35$  | 83.12 </br> $_0.24$  |
| Cresci-15  | 96.73 </br> $_0.30$ | 96.46 </br> $_0.18$  | 96.43 </br> $_0.19$  |
| MGTAB      | 85.84 </br> $_0.92$ | 83.27 </br> $_0.80$  | 85.81 </br> $_0.33$  |

GAT
| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 72.80 </br> $_0.11$ |  72.31 </br> $_0.27$ | 71.57 </br> $_0.88$  |
| Cresci-15  | 96.49 </br> $_0.15$ |  96.18 </br> $_0.30$ | 95.86 </br> $_0.39$  |
| MGTAB      | 84.46 </br> $_1.13$ |  80.47 </br> $_1.29$ | 79.35 </br> $_1.58$  |

OS-GNN (backbone GAT)
| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 82.49 </br> $_0.42$ | 82.30 </br> $_0.37$  | 82.41 </br> $_0.25$  |
| Cresci-15  | 96.65 </br> $_0.36$ | 96.38 </br> $_0.39$  | 96.35 </br> $_0.40$  |
| MGTAB      | 86.75 </br> $_0.74$ | 85.39 </br> $_0.71$  | 87.18 </br> $_0.50$  |

       





