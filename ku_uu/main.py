import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sgcn import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs
from utils import *
import numpy as np
import pandas as pd


def main():
    args = parameter_parser()
    # print a args table
    tab_printer(args)
    edges = read_graph(args)  
    np.set_printoptions(precision=5, suppress=True)

    all_drug_order, div_order = get_kk_ku_uu_order(edges["ncount"])
    test_sum_ku = np.zeros(10)
    test_sum_uu = np.zeros(10)
    for i in range(10):
        kk_edges, ku_edges, uu_edges = split_KK_KU_UU(all_drug_order, div_order, i)

        trainer = SignedGCNTrainer(args, edges)

        positive_edges = np.array([edge[0:3] for edge in kk_edges if edge[2] > 0])
        negative_edges = np.array([edge[0:3] for edge in kk_edges if edge[2] < 0])

        test_positive_edges = np.array([])
        test_negative_edges = np.array([])

        trainer.setup_dataset(list(positive_edges), list(test_positive_edges), list(negative_edges),
                              list(test_negative_edges),kk_edges,ku_edges,uu_edges)

        trainer.create_and_train_model(1e-9)
        print("trainer.logs['performance_ku'][-1]",trainer.logs['performance_ku'][-1])
        test_sum_ku = np.array(trainer.logs['performance_ku'][-1]) + test_sum_ku
        print("test_sum_ku",test_sum_ku)
        test_sum_uu = np.array(trainer.logs['performance_uu'][-1]) + test_sum_uu
        print("test_sum_uu",test_sum_uu)
    
    print("results：")
    print("ku:", test_sum_ku / 10)
    print("uu:", test_sum_uu / 10)


    


    

if __name__ == "__main__":
    main()
