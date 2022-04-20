import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sgcn import SignedGCNTrainer
from param_parser import parameter_parser
#from utils import tab_printer, read_graph, score_printer, save_logs
from utils import *
import numpy as np
import pandas as pd


def main():
    args = parameter_parser()
    tab_printer(args)
    set_random_seed(args.seed)
    edges = read_graph(args) 
    mode = "SGCN" # or "GCN" or "GAT"
    trainer = SignedGCNTrainer(args, edges,mode)
    np.set_printoptions(precision=5,suppress=True)
    pos_reorder, neg_reorder, pos_order, neg_order = get_cross_validation_dataset(edges, args, 42)
    
    # transductive learning
    for i in range(args.fold):
        test_positive_edges = np.array(edges["positive_edges"])[pos_order[i]]
        positive_edges = np.array(edges["positive_edges"])[list(set(pos_reorder).difference(set(pos_order[i])))]
        test_negative_edges = np.array(edges["negative_edges"])[neg_order[i]]
        negative_edges = np.array(edges["negative_edges"])[(list(set(neg_reorder).difference(set(neg_order[i]))))]
    
        trainer.setup_dataset(list(positive_edges), list(test_positive_edges), list(negative_edges),
                              list(test_negative_edges))
        # weight decay is 1e-9
        trainer.create_and_train_model(1e-9)
        if args.test_size > 0:
            #trainer.save_model()
            score_printer(trainer.logs)
            save_logs(args, trainer.logs)
    
    test_sum = np.zeros(len(trainer.logs['performance'][0]))
    for i in range(len(trainer.logs['performance'])-1):
        test_sum = np.array(trainer.logs['performance'][i + 1]) + test_sum
    trainer.logs['performance'].append(test_sum/args.fold)
    print("training over, results：")
    print(test_sum/args.fold)



    

if __name__ == "__main__":
    main()
