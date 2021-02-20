"""
Data Preprocessing
Preprocessing conll file into tsv
TO-DO: Get information about the data
"""

import os, sys, argparse

if __name__ == "__main__":
    main()

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
    # TODO: Step 1.3 - Get information about the data and write to .info
    return None