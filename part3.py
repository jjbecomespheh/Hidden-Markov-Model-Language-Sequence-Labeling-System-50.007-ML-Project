from part1 import *
from part2 import *
import pprint
from math import log
import itertools
import ast

def viterbi_5th_best(labels_transition_count, labels_emission_count, transition_count, emission_count, seq, k=1):
   
    """
    This approach is a modified version of the Viterbi algorithm to obtain the 5th best sequence.

    pi is a dictionary with length equal to the length of a sentence. If has keys numbering from 0 to length of sentence -1.
    The values of pi consists dictionaries with keys as the possible sequence labels for the words, and values as the probability for the sequence
    """

    pi = { 0 : {"[_, 'Start']": 0} }

    for j in range(1, len(seq)+1):

        pi[j] = {}
        temp_ls = []
        for v in labels_emission_count:
            for u in pi[j-1].keys():
                #Find all prob from prev layer to node v -> Then get the max
                new_u_pair = u.strip('][').split(', ')
                new_u = new_u_pair[-1][1:-1] # eg. remove "" from "'Start'"
                temp_inner_ls = []
                
                transition_prob = est_tr_params(transition_count, labels_transition_count, new_u, v)
                emission_prob = est_em_params_unk(emission_count, labels_emission_count, seq[j-1], v)
                
                for times in range(len(new_u_pair)):
                    temp_inner_ls.append(new_u_pair[times][1:-1])
                temp_inner_ls.append(v)

                if transition_prob > 0 and emission_prob > 0:
                    # Use max log likelihood to resolve numerical underflow issue
                    temp_inner_ls.append(pi[j-1][u]  + log(transition_prob) + log(emission_prob))
                else:
                    temp_inner_ls.append(float("-inf"))

                temp_ls.append(temp_inner_ls)

        temp_ls.sort(key = lambda x: -x[-1])
        temp_ls = temp_ls[:5]

        for i in range(len(temp_ls)):
            uv_ls = []
            for time in range(len(temp_ls[i])-1):
                uv_ls.append(temp_ls[i][time])
            uv = str(uv_ls)
            pi[j][uv] = temp_ls[i][-1]

    final_layer = pi[len(pi)-1]
    fifth_best = 100000

    for value in final_layer.values():
        
        if value < fifth_best:
            fifth_best = value

    final_labels = list(final_layer.keys())[list(final_layer.values()).index(fifth_best)]
    final_labels = ast.literal_eval(final_labels)[2:]
    return final_labels

def output_to_file(dev_path, output_path, labels_transition_count, labels_emission_count, transition_count, emission_count):
    with open(dev_path, "r") as f:
        open(output_path, "w")
        
        lines_ls = f.readlines()
        
        temp_sentence = []

        for i in range(len(lines_ls)):
            line = lines_ls[i].replace("\n", "")
            
            if line != "":
                temp_sentence.append(line)
            else:
                optimum_labels = viterbi_5th_best(labels_transition_count, labels_emission_count, transition_count, emission_count, temp_sentence)
                with open(output_path, "a") as f:
                    for i in range(len(temp_sentence)):
                        out = f"{temp_sentence[i]} {optimum_labels[i]}\n" 
                        f.write(out)
                    f.write('\n')
            
                temp_sentence = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="ES", help="ES or RU")

    opt = parser.parse_args()

    foldername = opt.folder
    
    train_path = f"./{foldername}/train"
    dev_path = f"./{foldername}/dev.in"
    output_path = f"./{foldername}/dev.p3.out"

    transition_count, labels_transition_count = count_transition(train_path)
    emission_count, labels_emission_count = count_emission(train_path)
  
    output_to_file(dev_path, output_path, labels_transition_count, labels_emission_count, transition_count, emission_count)