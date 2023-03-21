import torch
from torch import nn
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, SAGEConv, RGATConv
from layers import SMOTEGCNConv, SMOTESAGEConv, SMOTEGATConv, SMOTERGCNConv
import torch.nn.functional as F



class RGAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(RGAT, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgat1 = RGATConv(embedding_dimension, hidden_dimension, num_relations=relation_num)
        self.rgat2 = RGATConv(hidden_dimension, out_dim, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        # x = self.linear_relu_input(feature.to(torch.float32))
        x = self.rga1(feature, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgat2(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        # x = self.linear_output2(x)

        return x



class RGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn1(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class GAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.gat1 = GATConv(embedding_dimension, int(hidden_dimension / 8), heads=8)
        self.gat2 = GATConv(hidden_dimension, out_dim)


    def forward(self, feature, edge_index, edge_type):
        x = F.relu(self.gat1(feature, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return x



class GCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.gcn1 = GCNConv(embedding_dimension, hidden_dimension)
        self.gcn2 = GCNConv(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = F.relu(self.gcn1(feature, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        return x



class SMOTERGAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SMOTERGAT, self).__init__()
        self.dropout = dropout

        self.gat1 = SMOTEGATConv(768 * 2, int(hidden_dimension / 4), heads=4)
        self.gat2 = SMOTEGATConv(768 * 2, int(hidden_dimension / 4), heads=4)
        self.gat1_2 = SMOTEGATConv(hidden_dimension, hidden_dimension)
        self.gat2_2 = SMOTEGATConv(hidden_dimension, hidden_dimension)

        self.linear_output1 = nn.Linear(hidden_dimension, out_dim)
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        feature1 = torch.cat([feature[:, :768], feature[:, 768:768 + 768 * 1]], axis=1)
        feature2 = torch.cat([feature[:, :768], feature[:, 768 * 2:768 * 2 + 768 * 1]], axis=1)
        x1 = self.gat1(feature1, edge_index)
        x2 = self.gat2(feature2, edge_index)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.gat1_2(x1, edge_index)
        x2 = self.gat2_2(x2, edge_index)
        x = x1 + x2
        return x


class SMOTERGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SMOTERGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_input1 = nn.Sequential(
            nn.Linear(768*2, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_input2 = nn.Sequential(
            nn.Linear(768*2, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = SMOTERGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = SMOTERGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_output1 = nn.Linear(hidden_dimension, out_dim)
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        feature1 = torch.cat([feature[:, :768], feature[:, 768:768 + 768 * 1]], axis=1)
        feature2 = torch.cat([feature[:, :768], feature[:, 768*2:768*2 + 768 * 1]], axis=1)
        x1 = self.linear_relu_input1(feature1.to(torch.float32))
        x2 = self.linear_relu_input2(feature2.to(torch.float32))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.linear_output1(x1)
        # x = self.linear_relu_output1(x)
        x2 = self.linear_output2(x2)
        x=x1+x2
        return x



class SMOTEGAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SMOTEGAT, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gat1 = SMOTEGATConv(hidden_dimension, int(hidden_dimension / 8), heads=8)
        self.gat2 = SMOTEGATConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class SMOTEGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SMOTEGCN, self).__init__()
        self.dropout = dropout

        self.gcn1 = SMOTEGCNConv(embedding_dimension, hidden_dimension)
        self.gcn2 = SMOTEGCNConv(hidden_dimension, out_dim)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)


    def forward(self, feature, edge_index, edge_type):
        x = F.relu(self.gcn1(feature, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)

        return x


class SMOTESAGE(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SMOTESAGE, self).__init__()

        self.dropout = dropout
        self.sage1 = SMOTESAGEConv(embedding_dimension, hidden_dimension)
        self.sage2 = SMOTESAGEConv(hidden_dimension, out_dim)


    def forward(self, feature, edge_index, edge_type):
        x = F.relu(self.sage1(feature, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)

        return x
		
		
		