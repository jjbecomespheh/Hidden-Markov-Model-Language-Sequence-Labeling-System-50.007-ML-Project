# 50.007-Machine-Learning-Project

## Pre-requisites

Ensure that you have installed **Python 3.4** or above, and structure the files as follows:
    
```
    .
    ├── ES
    │   ├── dev.in
    │   ├── dev.out
    │   └── train
    ├── EvalScript
    │   ├── Instruction.txt
    │   ├── dev.out
    │   ├── dev.prediction
    │   └── evalResult.py
    ├── README.md
    ├── RU
    │   ├── dev.in
    │   ├── dev.out
    │   └── train
    ├── part1.py
    ├── part2.py
    ├── part3.py
    └── part4.py
```
## About

In this project, we are tasked to create our own sequence labelling model for informal texts using Hidden Markov Model (HMM) theory that we learned in class. The objective is to train a sequence labelling system using Spanish and Russian text datasets to predict tag sequences for new sentences in both languages.

There are 4 parts to this project:

- Part 1 implements a simple sentiment analysis system by estimating the emission parameters. The argmax of the observation is then computed, and the precision, recall, and F scores are calculated using the evaluation script.

- Part 2 incorporates the estimation of transition parameters as well as the application of the Viterbi algorithm to predict tag sequences. The evaluation script is then used to calculate the precision, recall, and F scores.

- Part 3 is employs a modified Virterbi algorithm to find the 5-th best output sequence. The evaluation script is then used to calculate the precision, recall, and F scores.

- Part 4 implements an improved version of the sentiment analysis system, by employing the Structured Perceptron algorithm used in NLP. Similarly, the evaluation script is then used to calculate the precision, recall, and F scores.

For more details, you can read our [report](https://docs.google.com/document/d/1_Y7q9cOCGsVJFRzZo6uA2LrqNvjs4E4-UD4fxL0V_ss/edit?usp=sharing)

## Part 1

Choose between the folders `ES` or `RU` and type:

    python3 part1.py -f <input folder>
    
The default argument is folder `ES`

To run ES:

    python3 part1.py -f ES 

To run RU:

    python3 part1.py -f RU 

This script will generate a dev.p1.out file.

## Part 2

Choose between the folders `ES` or `RU` and type:

    python3 part2.py -f <input folder>
    
The default argument is folder `ES`

To run ES:

    python3 part2.py -f ES 

To run RU:

    python3 part2.py -f RU 

This script will generate a dev.p2.out file.

## Part 3

Choose between the folders `ES` or `RU` and type:

    python3 part3.py -f <input folder>
    
The default argument is folder `ES`

To run ES:

    python3 part3.py -f ES 

To run RU:

    python3 part3.py -f RU 

This script will generate a dev.p3.out file.

## Part 4

Choose between the folders `ES` or `RU` and type:

    python3 part4.py -f <input folder> -m <mode to run>
    

The default arguments are folder `ES` and mode `dev`

To run on ES for dev set:
    
    python3 part4.py
    
To run on RU for dev set:
    
    python3 part4.py -f RU

To run on ES for test set:
    
    python3 part4.py -m test
    
To run on RU for test set:
    
    python3 part4.py -f RU - test
    

This script will generate either a `test.p4.out` or `dev.p4.out` file into the specified output file path depending on the mode selected.

## Evaluation

To evaluate the results of the output files, run `python ./EvalScript/evalResult.py dev.out dev.prediction`, where **dev.out** is the gold file and **dev.prediction** is the output you have predict.

## Results

Part 1 | ES | RU
----------------------- | --------- | ------
Entity in gold data     | 255       | 461
Entity in prediction    | 1733      | 2089
Correct Entity          | 205       | 335
Entity precision        | 0.1183    | 0.1604
Entity recall           | 0.8039    | 0.7267
Entity F                | 0.2062    | 0.2627
Correct Sentiment       | 113       | 136
Sentiment precision     | 0.0652    | 0.0651
Sentiment recall        | 0.4431    | 0.2950
Sentiment F             | 0.1137    | 0.1067

Part 2 | ES | RU
----------------------- | --------- | ------
Entity in gold data     | 255       | 461
Entity in prediction    | 644       | 505
Correct Entity          | 128       | 177
Entity precision        | 0.1988    | 0.3505
Entity recall           | 0.5020    | 0.3839
Entity F                | 0.2848    | 0.3665
Correct Sentiment       | 105       | 124
Sentiment precision     | 0.1630    | 0.2455
Sentiment recall        | 0.4118    | 0.2690
Sentiment F             | 0.2336    | 0.2567

Part 3 | ES | RU
----------------------- | --------- | ------
Entity in gold data     | 255       | 461
Entity in prediction    | 536       | 746
Correct Entity          | 106       | 168
Entity precision        | 0.1978    | 0.2252
Entity recall           | 0.4157    | 0.3644
Entity F                | 0.2680    | 0.2784
Correct Sentiment       | 72        | 101
Sentiment precision     | 0.1343    | 0.1354
Sentiment recall        | 0.2824    | 0.2191
Sentiment F             | 0.1820    | 0.1674

Part 4 | ES | RU
----------------------- | --------- | ------
Entity in gold data     | 255       | 461
Entity in prediction    | 229       | 472
Correct Entity          | 62        | 274
Entity precision        | 0.2707    | 0.5805
Entity recall           | 0.2431    | 0.5944
Entity F                | 0.2562    | 0.5864
Correct Sentiment       | 50        | 167
Sentiment precision     | 0.2183    | 0.3538
Sentiment recall        | 0.1961    | 0.3623
Sentiment F             | 0.2066    | 0.3580