# NNIA Project
Final Project, NNIA (Wintersemester 2020/21), Saarland University

## Table of contents
* [General Information](#general-information)
* [Data Preprocessing](#data-preprocessing)
* [Authors](#authors)

## General Information
TODO: environment

## Data Preprocessing

For the preprocessing step, first concatenate the data into a single .conll file, if needed. For example, if there are several files all with the suffix `.gold_conll` and at `data/sample`, run
```
$ cat data/sample/*.gold_conll > data/sample.conll
```

Then use `data_prep.py` to preprocess the data, which takes two arguments, a single input file and an output directory. For example, run
```
$ python data_prep.py data/sample.conll output
```
The script will output two files in the output directory, namely a `sample.tsv` containing only relevant information (word position, word, and POS tag) and a `sample.info` containing some basic info on the data.

## Authors
In alphabetical order:
* **Meng Li** - [limengnlp](https://github.com/limengnlp)
* **Peilu Lin** - []()
* **Siyu Tao** - [siyutao](https://github.com/siyutao)