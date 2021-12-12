import pprint
from part1 import *
from math import log
import numpy as np

def est_tr_params(transition_count, labels_count, yi_1, yi):
    return transition_count[yi_1][yi]/labels_count[yi_1]

def viterbi(labels_transition_count, labels_emission_count, transition_count, emission_count, seq, k=1):
    # STEP 1: Initialization
    pi = { 0 : {"Start": 0} }
    optimum_labels = []
    pp = pprint.PrettyPrinter(depth=6)

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
     
    return optimum_labels

def count_transition(path):
    transition_count = {
                        "Start": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0},
                        "O": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0}, 
                        "B-positive": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0}, 
                        "I-positive": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0}, 
                        "B-negative": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0}, 
                        "I-negative": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0}, 
                        "B-neutral": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0}, 
                        "I-neutral": {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0},
                        }
    labels_count = {"Start": 0, "O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0, "Stop": 0}
    pp = pprint.PrettyPrinter(depth=6)

    with open(path, "r") as f:
        lines_ls = f.readlines()
        line = lines_ls[0].replace("\n", "")
        _ , lbl = line.rsplit(" ",1)
        transition_count["Start"][lbl] += 1
        labels_count["Start"] += 1

        for line_num in range(len(lines_ls)):

            if line_num == (len(lines_ls)-1):
                break
            
            line = lines_ls[line_num].replace("\n", "")
            lineplusone = lines_ls[line_num+1].replace("\n", "") 
            current_ls = line.rsplit(" ", 1)
            next_ls = lineplusone.rsplit(" ", 1)

            if len(current_ls) == 1 and len(next_ls) == 1: # Check if double empty line aka End of File
                pass
            elif len(current_ls) > 1 and len(next_ls) > 1:
                word = current_ls[0]
                lbl = current_ls[1]
                lblplusone = next_ls[1]
                transition_count[lbl][lblplusone] += 1
                labels_count[lbl] += 1

            elif len(next_ls) == 1: # At last character
                transition_count[lbl]["Stop"] += 1
                labels_count[lbl] += 1

            elif len(current_ls) == 1: # At empty line
                transition_count["Start"][lblplusone] += 1
                labels_count["Start"] += 1
        labels_count["Stop"] = labels_count["Start"]   

    return transition_count, labels_count

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="./ES", help="Folder path")
    parser.add_argument("--output", "-o", type=str, default="./ES/dev.p2.out", help="Output file path")

    opt = parser.parse_args()

    foldername = opt.folder
    output_path = opt.output
    
    train_path = f"{foldername}/train"
    dev_path = f"{foldername}/dev.in"

    transition_count, labels_transition_count = count_transition(train_path)
    emission_count, labels_emission_count = count_emission(train_path)
 
    with open(dev_path, "r") as f:
        open(output_path, "w")
        
        lines_ls = f.readlines()
        
        temp_sentence = []

        for i in range(len(lines_ls)):
            line = lines_ls[i].replace("\n", "")
            
            if line != "":
                temp_sentence.append(line)
            else:
                optimum_labels = viterbi(labels_transition_count, labels_emission_count, transition_count, emission_count, temp_sentence)
                with open(output_path, "a") as f:
                    for i in range(len(temp_sentence)):
                        out = f"{temp_sentence[i]} {optimum_labels[i]}\n" 
                        f.write(out)
                    f.write('\n')
            
                temp_sentence = []