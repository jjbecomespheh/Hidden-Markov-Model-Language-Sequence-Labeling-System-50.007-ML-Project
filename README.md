# 50.007-Machine-Learning-Project

## Pre-requisites

Ensure that the structure for the files are as follows:
    
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

