"""SGCN runner."""
import os
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import trange
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool,SAGPooling
from torch_geometric.nn import GATConv, GINConv,LayerNorm, global_add_pool, BatchNorm
from torch_geometric.nn import DeepGCNLayer
from torch_geometric.data import Batch
import torch.utils.data as Data
from signedsageconvolution import SignedSAGEConvolutionBase, SignedSAGEConvolutionDeep,ListModule
from torch_geometric import data as DATA
from utils import *
from MolDataset import MolDataset
import pickle

class SSI_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
    
    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores= self.readout(data.x, data.edge_index, batch=data.batch)
        global_graph_emb = global_add_pool(att_x, att_batch)

        # data = max_pool_neighbor_x(data)
        return data, global_graph_emb

class SignedGraphConvolutionalNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """

    def __init__(self, device, args, X, mode="SGCN", dropout=0.9):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        self.agg_mode = mode
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        self.device = device
        self.X = X
        self.setup_layers()

    def setup_layers(self):
        """
        Adding Base Layers, Deep Signed GraphSAGE layers.
        Assing Regression Parameters if the model is not a single layer model.
        """
        self.nodes = range(self.X.shape[0])
        self.ini_dim = self.X.shape[1]
        self.neurons = self.args.layers
        self.layers = len(self.neurons)
        self.Linear = torch.nn.Linear(self.ini_dim, self.ini_dim)
        self.ReLU = nn.ReLU()
        if self.agg_mode=="SGCN":
            self.positive_base_aggregator = SignedSAGEConvolutionBase(self.ini_dim*2,self.neurons[0]).to(self.device)
            self.negative_base_aggregator = SignedSAGEConvolutionBase(self.ini_dim*2,self.neurons[0]).to(self.device)
            self.positive_aggregators = []
            self.negative_aggregators = []
            for i in range(1, self.layers):
                self.positive_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                           self.neurons[i]).to(self.device))
                self.negative_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                           self.neurons[i]).to(self.device))
            self.positive_aggregators = ListModule(*self.positive_aggregators)
            self.negative_aggregators = ListModule(*self.negative_aggregators)

        elif self.agg_mode=="GAT":
            self.gat1 = GATConv(self.ini_dim, 128)
            self.gat2 = GATConv(128, 128)
        elif self.agg_mode=="GCN":
            self.gcn1 = GCNConv(self.ini_dim, 128)
            self.gcn2 = GCNConv(128, 128)

    def calculate_mse_function(self, pred, labels):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, labels)
        return loss


    def forward(self, positive_edges, negative_edges, labels, label_mask):
        """
        Model forward propagation pass. Can fit deep and single layer SGCN models.
        """
        X = self.Linear(self.X)
        X = self.ReLU(X)
        #X = self.X
        if self.agg_mode=="SGCN":
            self.h_pos=[]
            self.h_neg=[]
            self.h_pos.append(torch.tanh(
                self.positive_base_aggregator(X, positive_edges)))
            self.h_neg.append(torch.tanh(
                self.negative_base_aggregator(X, negative_edges)))
            for i in range(1, self.layers):
                self.h_pos.append(torch.tanh(
                    self.positive_aggregators[i - 1](self.h_pos[i - 1], self.h_neg[i - 1], positive_edges, negative_edges)))
                self.h_neg.append(torch.tanh(
                    self.negative_aggregators[i - 1](self.h_neg[i - 1], self.h_pos[i - 1], positive_edges, negative_edges)))
            self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1) 
        elif self.agg_mode =="GAT":
            edge = torch.cat((positive_edges,negative_edges),1)
            X = self.gat1(X, positive_edges)
            X = F.normalize(X, p=2, dim=-1)
            self.z = self.gat2(X, positive_edges)
        elif self.agg_mode =="GCN":
            edge = torch.cat((positive_edges, negative_edges), 1)
            X= self.gcn1(X,positive_edges)
            X = F.normalize(X, p=2, dim=-1)
            self.z = self.gcn2(X,positive_edges)
        self.X_mol = F.normalize(self.z)
        pred = torch.flatten(torch.mm(self.X_mol, self.X_mol.t()) * label_mask)
        loss = self.calculate_mse_function(pred, labels)

        return loss, self.X_mol, pred


class SignedGCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """

    def __init__(self, args, edges, mode ="SGCN"):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure with positive and negative edges separated.
        :param mode : The aggregating method mode
        """
        self.args = args
        self.edges = edges
        self.agg_mode = mode
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance"] = [
            ["Epoch", "AUC", "F1", "corr", "msetotal", "mse1", "mse2", "mse5", "auroc", "precision1", "precision2",
             "precision5"]]
        self.logs["training_loss"] = [["Epoch", "loss"]]

    def setup_dataset(self, positive_edges, test_positive_edges, negative_edges, test_negative_edges):

        self.positive_edges = positive_edges  
        self.test_positive_edges = test_positive_edges 
        self.negative_edges = negative_edges 
        self.test_negative_edges = test_negative_edges
        
        # the whole edges number
        self.ecount = len(self.positive_edges + self.negative_edges)
        
        # the whole trainset of asociation graph
        self.train_edges = self.positive_edges + self.negative_edges

        # the whole testset of asociation graph
        self.test_edges = self.test_positive_edges + self.test_negative_edges

        # rebulit trainset, only get [drug1,drug2]
        self.positive_edges = np.array(np.array(self.positive_edges)[:, [0, 1]])
        self.negative_edges = np.array(np.array(self.negative_edges)[:, [0, 1]])
        
        # testset do not change
        self.test_positive_edges = np.array(self.test_positive_edges)
        self.test_negative_edges = np.array(self.test_negative_edges)

        # get trainset feature in association graph self.X:[2428, 300], the drug's features
        self.X = setup_features(self.args,
                                self.positive_edges,
                                self.negative_edges,
                                self.edges["ncount"])
        
        print("self.X",self.X.shape)
        self.X = torch.from_numpy(self.X).float().to(self.device)
     
        self.positive_edges = torch.from_numpy(np.array(self.positive_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)
        self.negative_edges = torch.from_numpy(np.array(self.negative_edges,
                                                        dtype=np.int64).T).type(torch.long).to(self.device)

        self.train_labels, self.train_mask = get_label_list(self.train_edges, self.edges["ncount"])
        self.train_labels = torch.from_numpy(self.train_labels).float().to(self.device)
        self.train_mask = torch.from_numpy(self.train_mask).float().to(self.device)

        self.test_labels, self.test_mask = get_label_list(self.test_edges, self.edges["ncount"])
        self.test_labels = torch.from_numpy(self.test_labels).float().to(self.device)
        self.test_mask = torch.from_numpy(self.test_mask).float().to(self.device)

    def score_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        self.model.eval()
       
        with torch.no_grad():
            loss, self.train_z, pred = self.model(self.positive_edges, self.negative_edges, self.test_labels,
                                              self.test_mask)
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(
            pred.cpu().detach().numpy(), self.test_labels.cpu().numpy(), self.edges['ncount'])
        self.logs["performance"].append(
            [epoch + 1, 0, 0, corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5])

    def create_and_train_model(self, weight_decay):
        """
        Model training and scoring.
        """
        print("\n SGCN Training started.\n")
        self.model = SignedGraphConvolutionalNetwork(self.device, self.args, self.X, self.agg_mode).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=weight_decay)
        self.model.train()
        print("model",self.model)
        self.epochs = trange(self.args.epochs, desc="Loss")
        for epoch in self.epochs:
            self.optimizer.zero_grad()
            loss, mol_x, pred = self.model(self.positive_edges, self.negative_edges, self.train_labels,
                                       self.train_mask)
            loss.backward()
            self.epochs.set_description(
                "SGCN (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                self.logs["training_loss"].append([epoch + 1, loss.item()])

        if self.args.test_size > 0:
            self.score_model(self.args.epochs-1)

    def save_model(self):
        """
        Saving the embedding and model weights.
        """
        if not os.path.exists(os.path.dirname(self.args.regression_weights_path)):
            os.mkdir(os.path.dirname(self.args.regression_weights_path))
        if not os.path.exists(os.path.dirname(self.args.embedding_path)):
            os.mkdir(os.path.dirname(self.args.embedding_path))    
        
        print("\nEmbedding is saved.\n")
        self.train_z = self.train_z.cpu().detach().numpy()
        embedding_header = ["id"] + ["x_" + str(x) for x in range(self.train_z.shape[1])]
        self.train_z = np.concatenate(
            [np.array(range(self.train_z.shape[0])).reshape(-1, 1), self.train_z], axis=1)
        self.train_z = pd.DataFrame(self.train_z, columns=embedding_header)
        self.train_z.to_csv(self.args.embedding_path, index=None)
        print("\nRegression weights are saved.\n")
        

def collate(data_list):
    batch = Batch.from_data_list([data for data in data_list])
    return batch
