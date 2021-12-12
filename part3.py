from part1 import *
from part2 import *
import pprint
from math import log

def viterbi_5th_best(labels_transition_count, labels_emission_count, transition_count, emission_count, seq, k=1):
     # STEP 1: Initialization
    pi = { 0 : {"Start": [[1,[]]]} }
    optimum_labels = []

    # STEP 2: Move Forward -> pi(j+1,u) = max{pi(j,v)*(b_u(x_j+1))*(a_v,u)}
    for j in range(1, len(seq)+1):
        pi[j] = {}
        outer_ls = []
        
        for v in labels_emission_count:
            prob = {}
            ls = []

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
                    for i in range(len(pi[j-1][u])):
                        prob[tuple(pi[j-1][u][i][1]+[u])] = pi[j-1][u][i][0] \
                        + log(transition_prob) \
                        + log(emission_prob)

                else:
                    for i in range(len(pi[j-1][u])):
                        prob[tuple(pi[j-1][u][i][1]+[u])] = float("-inf")

            probValues = sorted(prob.values(), reverse=True)
            probKeys = sorted(prob, key=prob.get, reverse=True)
            if len(probValues) == 1:
                pi[j][v] = [[probValues[0], list(probKeys[0])]]
            else:
                pi[j][v] = [[probValues[0], list(probKeys[0])], [probValues[1], list(probKeys[1])], [probValues[2], list(probKeys[2])]]



        pp = pprint.PrettyPrinter(depth=6)
        pp.pprint(pi)
    print(optimum_labels)     
    # optimum_labels = None   
    return optimum_labels

def viterbi_5th_best_copy(labels_transition_count, labels_emission_count, transition_count, emission_count, seq, k=1):
     # STEP 1: Initialization
    pi = { 0 : {"Start": 0} }
    optimum_labels = []

    # STEP 2: Move Forward -> pi(j+1,u) = max{pi(j,v)*(b_u(x_j+1))*(a_v,u)}
    for j in range(1, len(seq)+1):
        pi[j] = {}
        outer_ls = []
        
        for v in labels_emission_count:
            prob = {}
            ls = []

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

                ls.append(prob[u])

            if j == 1:
                pi[j][v] = prob[u]

            else:
                sorted_ls = sorted(ls)
                print(sorted_ls)
                fifth_best_utov = sorted_ls[-3]
                pi[j][v] = fifth_best_utov
            outer_ls.append(pi[j][v])
            
        sorted_outer_ls = sorted(outer_ls)
        fifth_best_v = sorted_outer_ls[-3]
        
        for key, value in pi[j].items():
            if value == fifth_best_v:
                optimum_labels.append(key)
                break

        pp = pprint.PrettyPrinter(depth=6)
        pp.pprint(pi)
    print(optimum_labels)     
    # optimum_labels = None   
    return optimum_labels

if __name__ == "__main__":
    foldername = "ES"
    output_file = "dev.p2.out"
    
    train_path = f"{foldername}/train"
    dev_path = f"{foldername}/dev.in"
    output_path = f"{foldername}/{output_file}"

    seq = ["La", "comida", "estuvo", "muy", "sabrosa", "."]
    transition_count, labels_transition_count = count_transition(train_path)
    emmision_count, labels_emmision_count = count_emission(train_path)
    optimum_labels = viterbi_5th_best(labels_transition_count, labels_emmision_count, transition_count, emmision_count, seq)
    
    # with open(dev_path, "r") as f:
    #     open(output_path, "w")
        
    #     lines_ls = f.readlines()
        
    #     temp_sentence = []

    #     for i in range(len(lines_ls)):
    #         line = lines_ls[i].replace("\n", "")
            
    #         if line != "":
    #             temp_sentence.append(line)
    #         else:
    #             optimum_labels = viterbi(labels_transition_count, labels_emmision_count, transition_count, emmision_count, temp_sentence)
    #             with open(output_path, "a") as f:
    #                 for i in range(len(temp_sentence)):
    #                     out = f"{temp_sentence[i]} {optimum_labels[i]}\n" 
    #                     f.write(out)
    #                 f.write('\n')
            
    #             temp_sentence = []