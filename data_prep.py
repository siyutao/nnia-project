"""
Data Preprocessing:
Preprocessing conll file into tsv and get information about the data
"""

import os, sys, argparse

def parse_arguments():
    # parse command line args with argparse
    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('-i', metavar='input_file', type=str,
                        help='path to the input .conll file',default='./data/ontonotes.conll')
    parser.add_argument('-o', metavar='output_path', type=str,
                        help='path of the output directory',default='./data/')
    parser.add_argument('--split', dest='split', action='store_true')
    parser.add_argument('--no-split', dest='split', action='store_false')
    parser.set_defaults(split=True)
    args = parser.parse_args()
    return args.i, args.o, args.split

def convert(conll_file, tsv_file):
    # convert standard conll file into three column tsv file that contains only
    # a) position, b) word, c) pos, separated by tab, sequences separated by #
    with open(conll_file, "r") as f, open(tsv_file, "w") as o:
        for line in f:
            line = line.strip()
            if line == "":
                # empty line marks end of a sentence
                o.write("*\n")
            elif line[0] != "#":
                # split line by tab and slice list
                split_line = line.split()[2:5]
                # output line rejoined with tab
                output_line = "\t".join(split_line)
                o.write(output_line)
                # new line
                o.write("\n")

def get_info(tsv_file, info_file):
    # Get information about the data and write to .info
    length=[]
    tag_count={}
    preline=''
    for line in open(tsv_file):
        line=line.strip('\n').split('\t')
        if line[0]=='*':
            l=int(preline[0])+1
            length.append(l)
        else:
            tag=line[2]
            if tag not in tag_count:
                tag_count[tag]=1
            else:
                tag_count[tag]+=1
        preline= line
    # Maximum sequence length;
    seq_max=max(length)
    # Minimum sequence length;
    seq_min=min(length)
    # Number of sequences;
    seq_num= len(length)
    # Mean sequence length;
    seq_mean=sum(length)/seq_num
    # List of tags and the percentage of the words that have these tags.
    tag_sum=sum(tag_count.values())
    for k,v in tag_count.items():
        tag_count[k]='{:.3%}'.format(v/tag_sum )
    
    # Output to .info
    with open(info_file,'w') as f:
        f.write(f'Maximum sequence length: {seq_max}\n')
        f.write(f'Minimum sequence length: {seq_min}\n')
        f.write(f'Mean sequence length: {seq_mean}\n')
        f.write(f'Number of sequences: {seq_num}\n\n')
        f.write(f'Tags:\n')
        for k,v in tag_count.items():
            f.write(f'{k:5}    {v}\n')
    return None

def write_to_tsv(examples, tsv_file):
    with open(tsv_file, "w") as writer:
        for sequence in examples:
            # each sequence
            for splits in sequence:
                output_line = "\t".join(splits)
                writer.write(output_line)
                writer.write("\n")
            # end of a sequence
            writer.write("*\n")

def convert_and_split(conll_file, outputdir):
    examples = []
    with open(conll_file, "r") as f:
        sequence = []
        for line in f.readlines():
            line = line.strip()
            if line == "":
                # save the current sequence
                examples.append(sequence)
                # clear for next sequence
                sequence = []
            elif line[0] != "#":
                # split line by tab and slice list
                splits = line.split()[2:5]
                sequence.append(splits)
    # train dev test split
    train_ratio = 0.8
    dev_ratio = 0.1
    test_ratio = 0.1
    train_index = int(len(examples) * train_ratio)
    dev_index = int(len(examples) * (1 - test_ratio))
    train_examples = examples[:train_index]
    dev_examples = examples[train_index:dev_index]
    test_examples = examples[dev_index:]
    # output
    write_to_tsv(train_examples, os.path.join(outputdir, 'train.tsv'))
    write_to_tsv(dev_examples, os.path.join(outputdir, 'dev.tsv'))
    write_to_tsv(test_examples, os.path.join(outputdir, 'test.tsv'))

def main():
    # parse the arguments
    conll_input, output_path, split = parse_arguments()
    # create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    if split:
        convert_and_split(conll_input, output_path)
    else:
        # output files (.tsv and .info)
        tsv_ouput = os.path.join(output_path, "data.tsv")
        info_output = os.path.join(output_path, "data.info")
        # convert .conll to .tsv, i.e. extract POS tags
        convert(conll_input, tsv_ouput)
        # get information about data and write to .info
        get_info(tsv_ouput, info_output)

if __name__ == "__main__":
    main()