def est_em_params(emissions, emissions_count, x,y):
    
    count_yx = emissions[y][x] 
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

    emissions = {"O":{}, "B-positive":{}, "I-positive":{}, "B-negative":{}, "I-negative":{}, "B-neutral":{}, "I-neutral":{}}
    emissions_count = {"O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0}

    with open(path, "r") as f:
        lines_ls = f.readlines()

        for line in lines_ls[:15]:
            line = line.replace("\n", "")
            temp_ls = line.split()

            if len(temp_ls) > 0:
                """ 
                Store the observation and its count as a dictionary 
                in the emissions dictionary 

                Eg. For label 0 and word disfrutemos, 
                {"0": {"disfrutemos: 1}} 

                """
                emissions[temp_ls[1]][temp_ls[0]] = emissions[temp_ls[1]].get(temp_ls[0], 0) + 1
                emissions_count[temp_ls[1]] += 1
            
    return emissions, emissions_count

def eval(foldername, emissions, emissions_count):
    filename = "dev.in"
    path = f"{foldername}/{filename}"

    emissions = {"O":{}, "B-positive":{}, "I-positive":{}, "B-negative":{}, "I-negative":{}, "B-neutral":{}, "I-neutral":{}}
    emissions_count = {"O": 0, "B-positive": 0, "I-positive": 0, "B-negative": 0, "I-negative": 0, "B-neutral": 0, "I-neutral": 0}

    with open(path, "r") as f:
        lines_ls = f.readlines()

        for line in lines_ls:
            line = line.replace("\n", "")
            temp_ls = line.split()

            if len(temp_ls) > 0:
                """ 
                Store the observation and its count as a dictionary 
                in the emissions dictionary 

                Eg. For label 0 and word disfrutemos, 
                {"0": {"disfrutemos: 1}} 

                """
                emissions[temp_ls[1]][temp_ls[0]] = emissions[temp_ls[1]].get(temp_ls[0], 0) + 1
                emissions_count[temp_ls[1]] += 1
            
    return emissions, emissions_count


if __name__ == "__main__":
    foldername = "ES"
    emissions, emissions_count = train(foldername)
    print(emissions_count)
    print(emissions)
    # emissions, emissions_count = eval(foldername, emissions, emissions_count)
    # print(emissions["O"]["calidad"])
    # print(emissions_count["O"])

    # em_param = est_em_params(emissions, emissions_count, "calidad", "O")
    # print(em_param)