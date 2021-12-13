import string
import pprint
import re
from part2 import *
import time
from math import log
from collections import Counter, defaultdict

def viterbi_modified(states, seq, weights):

    k = 1
    pi = {
        k: {}
    }

    pi_end = {
        k: {}
    }

    for state in states:
        features = get_features(seq[0], state, "Start")
        weights_of_features = sum((weights[x] for x in features))
        pi[k][state] = weights_of_features
        pi_end[k][state] = "Start"

    for observation in seq[1:]:
        k += 1
        pi[k] = {}
        pi_end[k] = {}
        for v in states:
            prob = {}
            for u in pi[k-1].keys():
                features = get_features(observation, v, u)
                weights_of_features = sum((weights[x] for x in features))

                prob[u] = pi[k-1][u] + weights_of_features
                
            max_state = max(prob, key=prob.get)
            pi[k][v] = prob[max_state]
            pi_end[k][v] = max_state
        
    
    prob = {}
    for u in pi[k].keys():
        weights_of_features = weights[(u, "Stop")]
        prob[u] = pi[k][u] + weights_of_features
    
    y_n = max(prob, key=prob.get)
    state_pred_r = [y_n]

    for n in reversed(range(1, k+1)):
        next_state = state_pred_r[-1]
        state_pred_r.append(pi_end[n][next_state])
    state_pred_r.reverse()
    
    return state_pred_r[1:]


def structured_perceptron(training_sentences_n_labels, states, epochs=10, l_rate=0.1):
    
    weights = defaultdict(float)
    total_weights = defaultdict(tuple)

    for epoch in range(epochs):
        
        print(f"| Epoch {epoch+1} |", end=" ")
        total = 0
        corr = 0
        for seq in training_sentences_n_labels:
            
            seq_words = [word_label_ls[0] for word_label_ls in seq]
            seq_states = [word_label_ls[1] for word_label_ls in seq]

            optimum_labels = viterbi_modified(states, seq_words, weights)

            gold_features = get_global_features(seq_words, seq_states)
            prediction_features = get_global_features(seq_words, optimum_labels)

            if optimum_labels != seq_states:
                weights, total_weights = update(weights, total_weights, gold_features, l_rate, epoch+1)
                for feature, count in prediction_features.items():
                    weights[feature] = weights[feature] - l_rate * count
            
            total += len(seq)
            corr += sum([1 for (predicted, gold) in zip(optimum_labels, seq_states) if predicted == gold])

        print(f"Training acc: {corr/total} |")

    return weights
            
def get_global_features(words, tags):
    counts = Counter()
    for i, (word, tag) in enumerate(zip(words, tags)):
        previous_tag = "Start" if i == 0 else tags[i-1]
        counts.update(get_features(word, tag, previous_tag))
    return counts

def update(weights, total_weights, features, l_rate, epochs):
    for feature, count in features.items():
        w = weights[feature]
        if not total_weights[feature]:
            w_iteration, total_weight = (0, 0)
        else:
            w_iteration, total_weight = total_weights[feature]

        total_weight += (epochs - w_iteration) * w
        w_iteration = epochs
        total_weight += l_rate * count

        weights[feature] += l_rate * count
        total_weights[feature] = (w_iteration, total_weight)
    return weights, total_weights

def get_features(word, state, prev_state):
    word_lower = word.lower()

    bi_prefix = word_lower[:2]
    tri_prefix = word_lower[:3]
    quad_prefix = word_lower[:4]
    penta_prefix = word_lower[:5]
    bi_suffix = word_lower[-2:]
    tri_suffix = word_lower[-3:]
    quad_suffix = word_lower[-4:]
    penta_suffix = word_lower[-4:]
    
    features = [
        f"bi_prefix_{bi_prefix}",
        f"bi_prefix+state_{bi_prefix}_{state}",
        f"bi_prefix+2states_{bi_prefix}_{prev_state}_{state}",
        f"bi_suffix_{bi_suffix}",
        f"bi_suffix+state_{bi_suffix}_{state}",
        f"bi_suffix+2states_{bi_suffix}_{prev_state}_{state}",

        f"tri_prefix_{tri_prefix}",
        f"tri_prefix+state_{tri_prefix}_{state}",
        f"tri_prefix+2states_{tri_prefix}_{prev_state}_{state}",
        f"tri_suffix_{tri_suffix}",
        f"tri_suffix+state_{tri_suffix}_{state}",
        f"tri_suffix+2states_{tri_suffix}_{prev_state}_{state}",
        
        f"quad_prefix_{quad_prefix}",
        f"quad_prefix+state_{quad_prefix}_{state}",
        f"quad_prefix+2states_{quad_prefix}_{prev_state}_{state}",
        f"quad_suffix_{quad_suffix}",
        f"quad_suffix+state_{quad_suffix}_{state}",
        f"quad_suffix+2states_{quad_suffix}_{prev_state}_{state}",

        f"penta_prefix_{penta_prefix}",
        f"penta_prefix+state_{penta_prefix}_{state}",
        f"penta_prefix+2states_{penta_prefix}_{prev_state}_{state}",
        f"penta_suffix_{penta_suffix}",
        f"penta_suffix+state_{penta_suffix}_{state}",
        f"penta_suffix+2states_{penta_suffix}_{prev_state}_{state}",
        
        f"lower+state_{word_lower}_{state}",
        f"upper_{word[0].isupper()}_{state}",
        
        f"state_{state}",
        f"states_{prev_state}_{state}",      
    ]
    return features

def predict(states,training_sentences, weights):
    predictions = []
    for seq in training_sentences:
        pred_label = viterbi_modified(states, seq, weights)
        predictions.append(list(zip(seq, pred_label)))
    return predictions

def get_training_sentences(train_path):
    with open(train_path, "r") as f:
        lines_ls = f.readlines()
        training_sentences_n_labels = []
        sentence_list = []
        for i in range(len(lines_ls)):

            line = lines_ls[i].replace("\n", "")
            ls = line.rsplit(" ",1)
            
            if len(ls) >=2:
                sentence_list.append(ls)
            else:
                # Reached an empty line
                if len(sentence_list) > 1:
                    training_sentences_n_labels.append(sentence_list)
                sentence_list = []
    
    return training_sentences_n_labels


def output_to_file(dev_path, output_path, states, weights):
    all_sentences = []

    with open(dev_path, "r") as f:
        open(output_path, "w")
        
        lines_ls = f.readlines()
        
        temp_sentence = []

        for i in range(len(lines_ls)):
            line = lines_ls[i].replace("\n", "")
            
            if line != "":
                temp_sentence.append(line)
            
            else:
                if len(temp_sentence) > 1:
                    all_sentences.append(temp_sentence)
                temp_sentence = []

        optimum_labels = predict(states, all_sentences, weights)

        with open(output_path, "a") as f:
            for sentence in optimum_labels:
                for word, label in sentence:
                    out = f"{word} {label}\n" 
                    f.write(out)
                f.write('\n')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="ES", help="ES or RU")
    parser.add_argument("--mode", "-m", type=str, default="dev", help="dev or test")

    opt = parser.parse_args()

    foldername = opt.folder
    mode = opt.mode
    
    train_path = f"./{foldername}/train"
    dev_path = f"./{foldername}/{mode}.in"
    output_path = f"./{foldername}/{mode}.p4.out"

    states = ["O", "B-positive", "I-positive", "B-negative", "I-negative", "B-neutral", "I-neutral"]
    
    training_sentences_n_labels = get_training_sentences(train_path)

    print("-------Training Started-------")
    weights = structured_perceptron(training_sentences_n_labels, states, epochs=10, l_rate=0.1)
    print("-------Training Ended-------")

    output_to_file(dev_path, output_path, states, weights)
    
