import os
import numpy as np
import argparse
from utils.paths import results_path

parser = argparse.ArgumentParser()
parser.add_argument('--filename_form', type=str, default=None)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--key', type=str, default='ARI')
args = parser.parse_args()

def isfloat(token):
    try:
        float(token)
        return True
    except ValueError:
        return False

vals = []
for i in range(args.num_runs):
    filename = args.filename_form.format(i+1)
    print(filename)

    with open(filename, 'r') as f:
        line= f.readline()

    tokens = [token.strip(',') for token in line.split(' ')]
    ind = tokens.index(args.key)
    while not isfloat(tokens[ind]):
        ind += 1

    vals.append(float(tokens[ind]))

print(np.round(np.mean(vals), 3), np.round(np.std(vals), 3))
