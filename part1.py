import argparse

def est_em_params(emission_count, labels_count, x,y):
    
    count_yx = emission_count[x][y] 
    count_y = labels_count[y]
    return count_yx/count_y

def est_em_params_unk(emission_count, labels_count, x,y, k=1):

    token= "#UNK#"

    if x not in emission_count:
        x = "#UNK#"
    
    if x == token:
        return k/ (labels_count[y]+k)

    else:
        count_yx= emission_count[x][y] 
        return count_yx/ (labels_count[y]+k)

def calc_argmax(emission_count, labels_count, x):
    
    em_prob = {}

    for label in labels_count:
        em_prob[label] = est_em_params_unk(emission_count, labels_count, x, label)

    return max(em_prob, key=em_prob.get)

def save_to_file(output_file, line, tag):

    output = f"{line} {tag}\n"
    if line == "":
        output = "\n"

    with open(output_file, "a+") as f:
        f.write(output)

def count_emission(path):

    emission_count = {}
    labels_count = {"O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0}

    with open(path, "r") as f:
        lines_ls = f.readlines()

        for line in lines_ls:
            line = line.replace("\n", "")

            try:
                word, lbl = line.rsplit(" ",1)

                emission_count[word] = emission_count.get(word,{})
                
                # Populating the inner dictionary by initializing count to 0 for each label
                for label in labels_count:
                    emission_count[word][label] = emission_count[word].get(label, 0)

                emission_count[word][lbl] = emission_count[word].get(lbl, 0) + 1
                labels_count[lbl] += 1

            except:
                # Reached anempty line in input file
                pass
            
    return emission_count, labels_count

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, default="ES", help="ES or RU")

    opt = parser.parse_args()

    foldername = opt.folder
    
    train_path = f"./{foldername}/train"
    dev_path = f"./{foldername}/dev.in"
    output_path = f"./{foldername}/dev.p1.out"

    emission_count, labels_emission_count = count_emission(train_path)

    with open(dev_path, "r") as f:
        open(output_path, "w")

        for line in f:

            line = line.replace("\n", "")

            if line != "":
                tag = calc_argmax(emission_count, labels_emission_count, line)
            else:
                tag = ""
            save_to_file(output_path, line, tag)