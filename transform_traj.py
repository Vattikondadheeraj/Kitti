import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='KITTI bin to npy conversion')
parser.add_argument('--file', type=str,      help='Location of the bin files')

args = parser.parse_args()


df =pd.read_csv(args.file)

col_list = list(df)
print(col_list)   # ['timestamp', 'x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w ']

# col_list[2], col_list[3] = col_list[3], col_list[2]
# col_list[5], col_list[6] = col_list[6], col_list[5]

df.to_csv(args.file)

# df.columns = col_list
# e