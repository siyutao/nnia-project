# NNIA Project
Final Project, NNIA (Wintersemester 2020/21), Saarland University

## Table of contents
* [General Information](#general-information)
* [Data Preprocessing](#data-preprocessing)
* [Authors](#authors)

## General Information
### Environment Setup
Create the environment from the environment.yml file:
```
$ conda env create -f environment.yml
```
Activate the project environment:
```
$ conda activate meng_peilu_siyu
```

## Data Preprocessing

For the preprocessing step, first concatenate the data into a single .conll file, if needed. For example, if there are several files all with the suffix `.gold_conll` and at `data/ontonotes-4.0`, run
```
$ cat data/ontonotes-4.0/*.gold_conll > data/ontonotes.conll
```

Then use `data_prep.py` to preprocess the data, which takes two arguments, a single input file and an output directory. For example, run
```
$ python data_prep.py -i data/ontonotes.conll -o data
```
The script will output two files in the output directory, namely a `data.tsv` containing only relevant information (word position, word, and POS tag) and a `data.info` containing some basic info on the data.

Run
```
$ python data_prep.py -i data/ontonotes.conll -o data/ontonotes_splits/ --split
```
to split the data into train, dev, test sets.

## Authors
In alphabetical order:
* **Meng Li** - [limengnlp](https://github.com/limengnlp)
* **Peilu Lin** - [palla-lin](https://github.com/palla-lin)
* **Siyu Tao** - [siyutao](https://github.com/siyutao)
