def est_em_params(emissions_count, states_count, x,y):
    
    count_yx = emissions_count[x][y] 
    count_y = states_count[y]
    return count_yx/count_y

def est_em_params_unk(emissions_count, states_count, x,y, k=1):

    token= "#UNK#"
    
    if x not in emissions_count:
        x = token
        emissions_count[x] = emissions_count.get(x,{})
        emissions_count[x][y] = emissions_count[x].get(y,0)
    
    if x == token:
        return k/ (states_count[y]+k)

    else:
        count_yx= emissions_count[x][y] 
        return count_yx/ (states_count[y]+k)

def calc_argmax(emissions_count, states_count, x):
    
    em_prob = {}
    for state in states_count:
        em_prob[state] = est_em_params_unk(emissions_count, states_count, x, state)

    return max(em_prob, key=em_prob.get)


def train(foldername):
    filename = "train"
    path = f"{foldername}/{filename}"

    emissions_count = {}
    states_count = {"O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0}

    with open(path, "r") as f:
        lines_ls = f.readlines()

        for line in lines_ls:
            line = line.replace("\n", "")

            try:
                """ 
                Store the observation and its count as a dictionary 
                in the emissions_count dictionary 

                Eg. For label 0 and word disfrutemos, 
                {"0": {"disfrutemos: 1}} 

                """
                word, label = line.split()
                emissions_count[word] = emissions_count.get(word,{})
                
                for state in states_count:
                    emissions_count[word][state] = emissions_count[word].get(state, 0)

                emissions_count[word][label] = emissions_count[word].get(label, 0) + 1
                states_count[label] += 1

            except:
                # empty space
                pass
            
    return emissions_count, states_count

def eval(foldername, emissions_count, states_count):
    filename = "dev.in"
    path = f"{foldername}/{filename}"

    pass


if __name__ == "__main__":
    foldername = "ES"
    emissions_count, states_count = train(foldername)
    test_word = "estuvo"
    tag = calc_argmax(emissions_count, states_count, test_word)
    print(states_count)
    print(emissions_count[test_word])
    print(tag)
    # em_param = est_em_params(emissions_count, states_count, "y", "O")
    # print(em_param)
    # emissions_count, states_count = eval(foldername, emissions_count, states_count)