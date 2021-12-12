import pprint
from part1 import *
from part2 import *
from math import log
import numpy as np

def fifth_best(labels_transition_count, labels_emmision_count, transition_count, emmision_count, seq):
    # STEP 1: Initialization
    pi = { 0 : {"Start": 0} }
    optimum_labels = []

    # STEP 2: Move Forward -> pi(j+1,u) = max{pi(j,v)*(b_u(x_j+1))*(a_v,u)}
    for j in range(1, len(seq)+1):
        pi[j] = {}
        for v in labels_emission_count:
            prob = {}
        
            for u in pi[j-1].keys(): #Find all prob from prev layer to node v -> Then get the max
                
                """
                j=1: u = "Start"

                {1: {O: max_value, B-positive: max_value , ...}}

                j=2: u = O, B-positive, ...

                {2: {O: max_value, B-positive: max_value , ...}}

                """

                transition_prob = est_tr_params(transition_count, labels_transition_count, u, v)
                emission_prob = est_em_params_unk(emission_count, labels_emission_count, seq[j-1], v)
                
                if transition_prob > 0 and emission_prob > 0:
                    # Use max log likelihood to resolve numerical underflow issue
                    prob[u] = pi[j-1][u] + log(transition_prob) + log(emission_prob)
                else:
                    prob[u] = float("-inf")
                    
            max_value = max(prob.values())
            pi[j][v] = max_value
        optimum_y = max(pi[j], key=pi[j].get)
        optimum_labels.append(optimum_y)
        pp = pprint.PrettyPrinter(depth=6)
        # pp.pprint(pi)
    #print(optimum_labels)        
    return optimum_labels


if __name__ == "__main__":
    foldername = "RU"
    output_file = "dev.p2.out"
    
    train_path = f"{foldername}/train"
    dev_path = f"{foldername}/dev.in"
    output_path = f"{foldername}/{output_file}"

    seq = ["La", "comida", "estuvo", "muy", "sabrosa", "."]
    transition_count, labels_transition_count = count_transition(train_path)
    emmision_count, labels_emmision_count = count_emission(train_path)
    optimum_labels = fifth_best(labels_transition_count, labels_emmision_count, transition_count, emmision_count, seq)
    