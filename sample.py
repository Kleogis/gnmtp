from model import GNMTP
from utils import *
import numpy as np
import os
import tensorflow as tf
import time
import argparse
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--num_iteration', help='num_iteration', default=10)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size', type=int, default=512)
parser.add_argument('--n_layer', help='number of layers', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=120)
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--target_prop', help='target properties', type=str)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument('--num_prop', help='number of properties', type=int, default=3)
parser.add_argument('--save_file', help='save file', type=str)
parser.add_argument('--prop_file', help='name of property file', type=str)
parser.add_argument('--result_filename', help='name of result filename', type=str, default='result.txt')
args = parser.parse_args()

_, _, char, vocab, _, _ = load_data(args.prop_file, args.seq_length)
vocab_size = len(char)

model = GNMTP(vocab_size,
             args
             )
model.restore(args.save_file)

target_prop = np.array([[float(p) for p in args.target_prop.split()] for _ in range(args.batch_size)])
start_codon = np.array([np.array(list(map(vocab.get, 'X')))for _ in range(args.batch_size)])

smiles = []
for _ in range(args.num_iteration):
    latent_vector = s = np.random.normal(args.mean, args.stddev, (args.batch_size, args.latent_size))
    generated = model.sample(latent_vector, target_prop, start_codon, args.seq_length)
    smiles += [convert_to_smiles(generated[i], char) for i in range(len(generated))]

print ('number of trial : ', len(smiles))
smiles = list(set([s.split('E')[0] for s in smiles]    ))
print ('number of generated smiles : ', len(smiles))
ms = [Chem.MolFromSmiles(s) for s in smiles]
ms = [m for m in ms if m is not None]
print ('number of valid smiles : ', len(ms))
with open(args.result_filename, 'w') as w:
    w.write('smiles\t MW\t LogP\t TPSA\n')
    for m in ms:
        try:
            w.write('%s\t%.3f\t%.3f\t%.3f\n' %(Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m)))
        except:
            continue            
