"""
Data Preprocessing:
Preprocessing conll file into tsv and get information about the data
"""

import os, sys, argparse

def parse_arguments():
    # parse command line args with argparse
    parser = argparse.ArgumentParser(description='Data Preprocessing')
    parser.add_argument('input', metavar ='input_file', type=str, help='path to the input .conll file')
    parser.add_argument('path', metavar='output_path' ,type=str, help='path of the output directory')
    args = parser.parse_args()
    return args.input, args.path

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

def main():
    # parse the arguments
    conll_input, output_path = parse_arguments()
    # create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    # output files (.tsv and .info)
    tsv_ouput = os.path.join(output_path, "sample.tsv")
    info_output = os.path.join(output_path, "sample.info")
    # convert .conll to .tsv, i.e. extract POS tags
    convert(conll_input, tsv_ouput)
    # get information about data and write to .info
    get_info(tsv_ouput, info_output)

if __name__ == "__main__":
    main()
