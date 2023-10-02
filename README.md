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

### Run models on bot detection datasets
````
python OS-GNN.py -dataset dataset -model model -smote smote
````
* **dataset**: including \[MGTAB, Twibot20, Cresci15\].  
* **model**: including \['GCN', 'GAT', 'SAGE', 'RGCN'\].  
* **smote**: including \[True, False\].  

e.g.
````
python OS-GNN.py -dataset MGTAB -model GCN -smote False
python OS-GNN.py -dataset MGTAB -model GCN -smote True
python OS-GNN.py -dataset Twibot20 -model GAT -smote True
python OS-GNN.py -dataset Cresci15 -model RGCN -smote True
````

### Run models on subgraph 
(different imbalanced ratio)
````
python subgraph-OS-GNN.py -dataset dataset -model model -smote smote -ratio ratio
````

* **ratio**: in the interval \[0, 1\]. 

e.g.
````
python subgraph-OS-GNN.py -dataset MGTAB -model GCN -smote False -ratio 0.05
python subgraph-OS-GNN.py -dataset Twibot20 -model GAT -smote False -ratio 0.20
````

### Run reweighting method
````
python GNN-reweight.py -dataset dataset -model model -reweight reweight -gamma gamma
````
* **reweight**: including \[CB, FL\].
* **smote**: including \[True, False\].  
* **beta**: parameter for CB loss.  (default = 0.9999)
* **gamma**: parameter for reweight. (default = 2.0)
* **alpha**: parameter for FocaL loss. (default = 0.5)

e.g.
````
python GNN-reweight.py -dataset MGTAB -model GCN -reweight CB --beta 0.99
python GNN-reweight.py -dataset MGTAB -model GCN -reweight FL --alpha 0.4
python GNN-reweight.py -dataset Twibot20 -model GCN -reweight FL --alpha 0.8
python GNN-reweight.py -dataset Cresci15 -model GCN -reweight FL --alpha 0.6
````



# Results
GCN

| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 68.76 </br> $_{0.60}$ |  68.30 </br> $_{0.51}$ | 68.29 </br> $_{0.62}$  |
| Cresci-15  | 96.50 </br> $_{0.36}$ |  96.20 </br> $_{0.42}$ | 95.95 </br> $_{0.53}$  |
| MGTAB      | 82.69 </br> $_{0.76}$ |  74.85 </br> $_{1.32}$ | 72.32 </br> $_{1.29}$  |
     

OS-GNN (backbone GCN)

| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 83.44 </br> $_{0.40}$ | 83.18 </br> $_{0.35}$  | 83.12 </br> $_{0.24}$  |
| Cresci-15  | 96.73 </br> $_{0.30}$ | 96.46 </br> $_{0.18}$  | 96.43 </br> $_{0.19}$  |
| MGTAB      | 85.84 </br> $_{0.92}$ | 83.27 </br> $_{0.80}$  | 85.81 </br> $_{0.33}$  |

GAT

| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 72.80 </br> $_{0.11}$ |  72.31 </br> $_{0.27}$ | 71.57 </br> $_{0.88}$  |
| Cresci-15  | 96.49 </br> $_{0.15}$ |  96.18 </br> $_{0.30}$ | 95.86 </br> $_{0.39}$  |
| MGTAB      | 84.46 </br> $_{1.13}$ |  80.47 </br> $_{1.29}$ | 79.35 </br> $_{1.58}$  |

OS-GNN (backbone GAT)

| Dataset    | Accuracy         | F1-macro          | Balanced accuracy |
| -----------| -----------------| ----------------- |-------------------|
| TwiBot-20  | 82.49 </br> $_{0.42}$ | 82.30 </br> $_{0.37}$  | 82.41 </br> $_{0.25}$  |
| Cresci-15  | 96.65 </br> $_{0.36}$ | 96.38 </br> $_{0.39}$  | 96.35 </br> $_{0.40}$  |
| MGTAB      | 86.75 </br> $_{0.74}$ | 85.39 </br> $_{0.71}$  | 87.18 </br> $_{0.50}$  |

       
# Dataset

For TwiBot-20, please visit the [Twibot-20 github repository](https://github.com/BunsenFeng/TwiBot-20).
For MGTAB please visit the [MGTAB github repository](https://github.com/GraphDetec/MGTAB).
For Cresci-15 please visit the [Twibot-20 github repository](https://github.com/GraphDetec/MGTAB).

We also offer the processed data set: [Cresci-15](https://drive.google.com/uc?export=download&id=13J-UkHZ6tuZedOI0RUgEoHiMIJRGAdNC), [MGTAB](https://drive.google.com/uc?export=download&id=1XfLYIz4M3KPnVpsEUwRMddSs548y29a5), [Twibot-20](https://drive.google.com/uc?export=download&id=1VtpWZzzRyze_5xIy2f1T6jV5lzyj1Oc9).
