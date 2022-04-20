from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
import torch
import numpy as np

class MolDataset(InMemoryDataset):
    def __init__(self, data_list_mol):
        self.data_mol = data_list_mol

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx]
        
class MolDataset_new(InMemoryDataset):
    def __init__(self, drug1_mol, drug2_mol, scores,device):
        self.drug1_mol = drug1_mol
        self.drug2_mol = drug2_mol
        self.scores = scores
        self.device= device

    def __len__(self):
        return len(self.drug1_mol)

    def __getitem__(self, idx):
        return (self.drug1_mol[idx], self.drug2_mol[idx], self.scores[idx])
    
    def collate(self, batchs):
        drug1_list=[]
        drug2_list=[]
        score=[]
        for batch in batchs:
            drug1 = batch[0]
            drug1_list.append(drug1)
            drug2 = batch[1]
            drug2_list.append(drug2)
            score.append(batch[2])
        drug1_batch = Batch.from_data_list(drug1_list) 
        drug2_batch = Batch.from_data_list(drug2_list)
        score = torch.from_numpy(np.array(score)).float().to(self.device)
        return drug1_batch, drug2_batch, score.cuda()
