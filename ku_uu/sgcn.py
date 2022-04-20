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
import pickle


class SignedGraphConvolutionalNetwork(torch.nn.Module):
    """
    Signed Graph Convolutional Network Class.
    For details see: Signed Graph Convolutional Network.
    Tyler Derr, Yao Ma, and Jiliang Tang ICDM, 2018.
    https://arxiv.org/abs/1808.06354
    """

    def __init__(self, device, args, X, X_from_mol_pretrain=None, dropout=0.9):
        super(SignedGraphConvolutionalNetwork, self).__init__()
        """
        SGCN Initialization.
        :param device: Device for calculations.
        :param args: Arguments object.
        :param X: Node features.
        """
        self.args = args
        torch.manual_seed(self.args.seed)
        self.device = device
        self.X = X
        self.setup_layers()
                
    def setup_layers(self):
        self.ini_dim = self.X.shape[1]
        self.nodes = range(self.X.shape[0])
        self.neurons = self.args.layers
        self.layers = len(self.neurons)

        self.positive_base_aggregator = SignedSAGEConvolutionBase( self.ini_dim*2,self.neurons[0]).to(self.device)
        self.negative_base_aggregator = SignedSAGEConvolutionBase( self.ini_dim*2,self.neurons[0]).to(self.device)
      
        self.positive_aggregators = []
        self.negative_aggregators = []
        for i in range(1, self.layers):
            self.positive_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                       self.neurons[i]).to(self.device))

            self.negative_aggregators.append(SignedSAGEConvolutionDeep(3 * self.neurons[i - 1],
                                                                       self.neurons[i]).to(self.device))

        self.positive_aggregators = ListModule(*self.positive_aggregators)
        self.negative_aggregators = ListModule(*self.negative_aggregators)
        self.linear= nn.Linear( self.ini_dim, self.ini_dim)
        self.relu = nn.ReLU()

    def calculate_mse_function(self, pred, labels):
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, labels)
        return loss


    def forward(self, X, positive_edges, negative_edges, labels, label_mask, mode):
        
        X = self.linear(X)
        X = self.relu(X)
        self.out = X        
                    
        self.h_pos, self.h_neg = [], []
        self.h_pos.append(torch.tanh(
            self.positive_base_aggregator(X, positive_edges)))
        self.h_neg.append(torch.tanh(
            self.negative_base_aggregator(X, negative_edges)))

        for i in range(1, self.layers):
            self.h_pos.append(torch.tanh(
                self.positive_aggregators[i - 1](self.h_pos[i - 1], self.h_neg[i - 1], positive_edges, negative_edges)))
            self.h_neg.append(torch.tanh(
                self.negative_aggregators[i - 1](self.h_neg[i - 1], self.h_pos[i - 1], positive_edges, negative_edges)))
        
        self.z = torch.cat((self.h_pos[-1], self.h_neg[-1]), 1) #[2428,128]


        self.X_mol = F.normalize(self.z)
        pred = torch.flatten(torch.mm(self.X_mol, self.X_mol.t()) * label_mask)
        loss = self.calculate_mse_function(pred, labels)
        if mode == "get_embed":
            return loss, self.out , pred, self.X_mol
        elif mode == "train":
            return loss, self.X_mol, pred


class SignedGCNTrainer(object):
    """
    Object to train and score the SGCN, log the model behaviour and save the output.
    """

    def __init__(self, args, edges):
        """
        Constructing the trainer instance and setting up logs.
        :param args: Arguments object.
        :param edges: Edge data structure with positive and negative edges separated.
        """
        self.args = args
        self.edges = edges
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print("device",self.device)
        self.setup_logs()

    def setup_logs(self):
        """
        Creating a log dictionary.
        """
        self.logs = {}
        self.logs["parameters"] = vars(self.args)
        self.logs["performance_ku"] = [
            ["Epoch", "corr", "msetotal", "mse1", "mse2", "mse5", "auroc", "precision1", "precision2",
             "precision5"]]
        self.logs["performance_uu"] = [
            ["Epoch", "corr", "msetotal", "mse1", "mse2", "mse5", "auroc", "precision1", "precision2",
             "precision5"]]
        self.logs["training_loss"] = [["Epoch", "loss"]]

    def setup_dataset(self, positive_edges, test_positive_edges, negative_edges, test_negative_edges, kk_edges,ku_edges,uu_edges):
        """
        Creating train and test split.
        """
        # get the trainset of association graph and testset of asociation graph
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

        self.kk_edges = kk_edges
        self.ku_edges = ku_edges
        self.uu_edges = uu_edges

        self.train_drugs = []
        for pair in kk_edges:
            if pair[0] not in self.train_drugs:
                self.train_drugs.append(int(pair[0]))
            if pair[1] not in self.train_drugs:
                self.train_drugs.append(int(pair[1]))
        self.test_drugs = []
        for i in range(2428):
            if i not in self.train_drugs:
                self.test_drugs.append(i)
        assert len(self.train_drugs)+len(self.test_drugs)==2428


        # rebulit trainset, only get [drug1,drug2]
        self.positive_edges = np.array(np.array(self.positive_edges)[:, [0, 1]])
        self.negative_edges = np.array(np.array(self.negative_edges)[:, [0, 1]])
        
        # testset do not change
        self.test_positive_edges = np.array(self.test_positive_edges)
        self.test_negative_edges = np.array(self.test_negative_edges)

        # get trainset feature in association graph self.X:[2428, 300], the drug's features
        X_train = setup_features(self.args, self.train_drugs)
        X_test = setup_features(self.args)
        
        self.X_train = torch.from_numpy(X_train).float().to(self.device)
        self.X_test = torch.from_numpy(X_test).float().to(self.device)
        
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


    def get_embed_model(self, epoch):
        """
        Score the model on the test set edges in each epoch.
        :param epoch: Epoch number.
        """
        self.model.eval()
        with torch.no_grad():
            loss, linear_embedding, pred, sgcn_embedding = self.model(self.X_test, self.positive_edges, self.negative_edges, self.test_labels,
                                                  self.test_mask, "get_embed")
        sgcn_embedding = sgcn_embedding.cpu().detach().numpy()
        jasscard_sim = self.get_CommonNeighbours_Similarity (linear_embedding).cpu().detach().numpy()
        unseen_sim = np.zeros([jasscard_sim.shape[0], jasscard_sim.shape[1]]) 
        unseen_sim[:, self.train_drugs] = jasscard_sim[:, self.train_drugs]  

        sort_unseen_sim_index = (-unseen_sim).argsort()[:, 0:10]  # get unseen drugs top 10 sims [2428,10]
        sort_unseen_sim_score = np.sort(-unseen_sim)[:, 0:10]  # get unseen drugs top 10 sims [2428,10]
        unseen_drugs_embed_from_sgcn = sgcn_embedding[sort_unseen_sim_index]  # [2428,10,128]
        unseen_drugs_embed_from_sgcn = np.mean(unseen_drugs_embed_from_sgcn, axis=1)  # [2428, 128]

        va_pred = []
        va_labels = []
        for pair in self.ku_edges:
            pair1 = int(pair[0])
            pair2 = int(pair[1])
            if pair1 in self.train_drugs:
                embedding1 = sgcn_embedding[pair1]
            else:
                embedding1 = unseen_drugs_embed_from_sgcn[pair1]

            if pair2 in self.train_drugs:
                embedding2 = sgcn_embedding[pair2]
            else:
                embedding2 = unseen_drugs_embed_from_sgcn[pair2]
            pred = np.matmul(embedding1, embedding2.T)
            va_pred.append(pred)
            va_labels.append(pair[2])

        te_pred = []
        te_labels = []
        for pair in self.uu_edges:
            pair1 = int(pair[0])
            pair2 = int(pair[1])
            embedding1 = unseen_drugs_embed_from_sgcn[pair1]
            embedding2 = unseen_drugs_embed_from_sgcn[pair2]
            pred = np.matmul(embedding1, embedding2.T)
            te_pred.append(pred)
            te_labels.append(pair[2])

        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5, precision = evaluation_new(
            va_pred, va_labels)
        self.logs["performance_ku"].append(
            [epoch + 1, corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5])
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5, precision = evaluation_new(
            te_pred, te_labels)
        self.logs["performance_uu"].append(
            [epoch + 1, corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5])


    def get_Jaccard_Similarity(self, interaction_matrix):
        X = interaction_matrix
        E = torch.ones_like(X.T)
        denominator = torch.mm(X, E) + torch.mm(E.T, X.T) - torch.mm(X, X.T)
        denominator_zero_index = torch.where(denominator == 0)
        denominator[denominator_zero_index] = 1
        result = torch.div(torch.mm(X, X.T), denominator)
        result[denominator_zero_index] = 0
        result = result - torch.diag(torch.diag(result))
        return result
    
    def get_Cosin_Similarity(self, interaction_matrix):
        X = interaction_matrix
        alpha = torch.sum(torch.mul(X,X),dim=1,keepdim=True)
        norm=torch.mm(alpha,alpha.T)
        index=torch.where(norm== 0)
        norm[index]=1
        similarity_matrix = torch.div( torch.mm(X , X.T) , (torch.sqrt(norm)))
        similarity_matrix[index]=0
        result=similarity_matrix
        result = result - torch.diag(torch.diag(result))
        return result
        

    def create_and_train_model(self, weight_decay,fold_idx=None):
        """
        Model training and scoring.
        """
        print("\n SGCN Training started.\n")
        self.model = SignedGraphConvolutionalNetwork(self.device, self.args, self.X_train).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=weight_decay)
        self.model.train()
        print("self.model",self.model)
        self.epochs = trange(self.args.epochs, desc="Loss")
        Mol_emb=0
        for epoch in self.epochs:
            self.optimizer.zero_grad()
            loss, mol_x, pred = self.model(self.X_train, self.positive_edges, self.negative_edges, self.train_labels,
                                       self.train_mask, "train")
            loss.backward()
            self.epochs.set_description(
                "SGCN (Loss=%g)" % round(loss.item(), 4))
            self.optimizer.step()
            if (epoch + 1) % 100 == 0:
                self.logs["training_loss"].append([epoch + 1, loss.item()])

        if self.args.get_embed:
            self.get_embed_model(self.args.epochs-1)



def collate(data_list):
    batch = Batch.from_data_list([data for data in data_list])
    return batch
