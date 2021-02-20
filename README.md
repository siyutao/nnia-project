# NNIA Project
Final Project, NNIA (Wintersemester 2020/21), Saarland University

## Table of contents
* [General Information](#general-information)
* [Data Preprocessing](#data-preprocessing)
* [Authors](#authors)

## General Information
TODO: environment

## Data Preprocessing

For the preprocessing step, first concatenate the dia into a single .conll file, if needed. For example, if the files are with the suffix `.gold_conll` and at `data/sample`, run
```
$ cat data/sample/*.gold_conll > data/sample.conll
```

Then use data_prep.py to preprocess the data, it takes two arguments, a single input file and the output directory. For example, run
```
$ python data_prep.py data/sample.conll output
```
The script will output two files into the output directory, `sample.tsv` containing only word position, word, POS tag, and `sample.info` containing some basic info on the data.

## Authors
In alphabetical order:
* **Meng Li** - [limengnlp](https://github.com/limengnlp)
* **Peilu Lin** - []()
* **Siyu Tao** - [siyutao](https://github.com/siyutao)