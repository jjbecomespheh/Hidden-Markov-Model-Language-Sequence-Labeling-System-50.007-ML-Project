def est_em_params(emissions, emissions_count, x,y):
    
    count_yx = emissions[x][y] 
    count_y = emissions_count[y]
    return count_yx/count_y

# def est_em_params_unk(emissions, emissions_count, x,y):

#     token= "#UNK#"

#     if emissions[y][x] 
    
#     count_yx = emissions[y][x] 
#     count_y = emissions_count[y]
#     return count_yx/count_y

def train(foldername):
    filename = "train"
    path = f"{foldername}/{filename}"

    emissions = {}
    emissions_count = {"O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0}

    with open(path, "r") as f:
        lines_ls = f.readlines()

        for line in lines_ls:
            line = line.replace("\n", "")

            try:
                """ 
                Store the observation and its count as a dictionary 
                in the emissions dictionary 

                Eg. For label 0 and word disfrutemos, 
                {"0": {"disfrutemos: 1}} 

                """
                word, label = line.split()
                emissions[word] = emissions.get(word,{})
                emissions[word][label] = emissions[word].get(label, 0) + 1
                emissions_count[label] += 1

            except:
                # empty space
                pass
            
    return emissions, emissions_count

def eval(foldername, emissions, emissions_count):
    filename = "dev.in"
    path = f"{foldername}/{filename}"

    pass


if __name__ == "__main__":
    foldername = "ES"
    emissions, emissions_count = train(foldername)
    print(emissions_count)
    print(emissions["y"])

    em_param = est_em_params(emissions, emissions_count, "y", "O")
    print(em_param)
    # emissions, emissions_count = eval(foldername, emissions, emissions_count)