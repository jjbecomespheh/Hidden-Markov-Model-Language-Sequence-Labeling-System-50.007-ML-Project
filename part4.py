import string
import pprint
import re
from part2 import *
import time
from math import log
from collections import Counter, defaultdict

def viterbi_algo(labels_emission_count, seq, weights):

    # Base Case
    k = 1
    pi = {
        k: {}
    }
    pi_edge = {
        k: {}
    }

    for state in labels_emission_count:
        features = get_features(seq[0], state, "START")
        feature_weights = sum((weights[x] for x in features))
        pi[k][state] = feature_weights
        pi_edge[k][state] = "START"

    # Move forward recursively
    for observation in seq[1:]:
        k += 1
        pi[k] = {}
        pi_edge[k] = {}
        for v in labels_emission_count:
            probabilities = {}
            for u in pi[k-1].keys():
                features = get_features(observation, v, u)
                feature_weights = sum((weights[x] for x in features))

                probabilities[u] = pi[k-1][u] + feature_weights
                
            max_state = max(probabilities, key=probabilities.get)
            pi[k][v] = probabilities[max_state]
            pi_edge[k][v] = max_state
        
    
    # Transition to STOP
    probabilities = {}
    for u in pi[k].keys():
        feature_weights = weights[(u, "STOP")]
        probabilities[u] = pi[k][u] + feature_weights
    
    # Best y_n
    y_n = max(probabilities, key=probabilities.get)
    state_pred_r = [y_n]

    # Backtrack
    for n in reversed(range(1, k+1)):
        next_state = state_pred_r[-1]
        state_pred_r.append(pi_edge[n][next_state])
    state_pred_r.reverse()
    
    return state_pred_r[1:]


def structured_perceptron(training_sentences_n_labels, transition_count, labels_transition_count, emission_count, labels_emission_count, epochs=10, l_rate=0.1):
    states = ["Start", "O", "B-positive", "I-positive", "B-negative", "I-negative", "B-neutral", "I-neutral", "Stop"]
    weights = defaultdict(float)
    total_weights = defaultdict(tuple)

    for epoch in range(epochs):
        total = correct = 0
        
        for seq in training_sentences_n_labels:
            
            seq_words = [word_label_ls[0] for word_label_ls in seq]
            seq_states = [word_label_ls[1] for word_label_ls in seq]

            optimum_labels = viterbi_algo(labels_emission_count, seq_words, weights)

            gold_features = get_global_features(weights, total_weights,seq_words, seq_states)
            prediction_features = get_global_features(weights, total_weights,seq_words, optimum_labels)

            if optimum_labels != seq_states:
                weights, total_weights = update(weights, total_weights, gold_features, l_rate, epoch+1)
                for feature, count in prediction_features.items():
                    weights[feature] = weights[feature] - l_rate * count
            
            total += len(seq)
            correct += sum([1 for (predicted, gold) in zip(optimum_labels, seq_states) if predicted == gold])

        print(f"| Epoch {epoch+1} | Training accuracy: {correct/total} |")

    return weights
            
def get_global_features(weights, total_weights, words, tags):
    counts = Counter()
    for i, (word, tag) in enumerate(zip(words, tags)):
        previous_tag = "Start" if i == 0 else tags[i-1]
        counts.update(get_features(word, tag, previous_tag))
    return counts

def update(weights, total_weights, features, l_rate, epochs):
    for f, count in features.items():
        w = weights[f]
        if not total_weights[f]:
            w_iteration, total_weight = (0, 0)
        else:
            w_iteration, total_weight = total_weights[f]
        # Update weight sum with last registered weight since it was updated
        total_weight += (epochs - w_iteration) * w
        w_iteration = epochs
        total_weight += l_rate * count

        # Update weight and total
        weights[f] += l_rate * count
        total_weights[f] = (w_iteration, total_weight)
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

def predict(labels_emission_count,training_sentences, weights):
    predictions = []
    for seq in training_sentences:
        pred_label = viterbi_algo(labels_emission_count, seq, weights)
        predictions.append(list(zip(seq, pred_label)))
    return predictions

def output(output_file, predictions):
    output = ""
    for seq in predictions:
        for entity in seq:
            output += "{}\n".format(" ".join(entity))
        output += "\n"
    output += "\n"
    with open(output_file, "w") as f:
        f.write(output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="./ES", help="Folder path")
    parser.add_argument("--output", "-o", type=str, default="./ES/dev.p4.out", help="Output file path")

    opt = parser.parse_args()

    foldername = opt.folder
    output_path = opt.output
    
    train_path = f"{foldername}/train"
    dev_path = f"{foldername}/dev.in"

    transition_count, labels_transition_count = count_transition(train_path)
    emission_count, labels_emission_count = count_emission(train_path)
    
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
                sentence_list = sentence_list[:-1]
                if len(sentence_list) > 1:
                    training_sentences_n_labels.append(sentence_list)
                sentence_list = []

    pp = pprint.PrettyPrinter(depth=6)
    weights = structured_perceptron(training_sentences_n_labels, transition_count, labels_transition_count, emission_count, labels_emission_count, epochs=20, l_rate=0.1)

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

        optimum_labels = predict(labels_emission_count, all_sentences, weights)

        with open(output_path, "a") as f:
            for sentence in optimum_labels:
                for word, label in sentence:
                    out = f"{word} {label}\n" 
                    f.write(out)
                f.write('\n')
