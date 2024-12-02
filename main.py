#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import json
import os
import uuid
import sys

import warnings
import gzip
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
warnings.simplefilter('ignore', PDBConstructionWarning)

print('Loading ML libraries...')
import torch
import torch as ch
import torch.nn as nn
from fastargs import Param, Section
from fastargs.validation import And, OneOf
import numpy as np
import src.config_parse_utils as config_parse_utils
from src.eval_utils import evaluate_model
from src.trainer import LightWeightTrainer
from src.models_and_optimizers import create_clip_model, load_model
import src.dist_utils as dist_utils
import src.data_utils as data_utils
from transformers import EsmTokenizer
import src.loader as loaders_utils
import webdataset as wds
from tqdm import tqdm
import tensorflow as tf
import os
import logging
from functools import partial
import src.loader as loader_utils
import src.zipdataset as zipdataset_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

print('Loading CLIP and ESM libraries...')
import src.models_and_optimizers as model_utils
from types import SimpleNamespace
from clip_main import get_wds_loaders
from utils import *
from torch.cuda.amp import autocast
from transformers import EsmTokenizer, EsmModel
import esm as esmlib


ROOT = "/home/gridsan/lguan/keating/rla/model_weights"


### WDS helpers

def process_residue(residue):
    atoms = ['N', 'CA', 'C', 'O']
    coordinates = []
    for r in atoms:
        coord = residue.child_dict.get(r, None)
        if coord is None:
            if r == 'O':
                coord = residue.child_dict.get('OXT', None)
            if coord is None:
                return None, None
        coordinates.append(np.array(coord.get_coord()))
    return np.stack(coordinates), seq1(residue.resname)

def process_chain(chain):
    coordinates = []
    seq = []
    for r in chain:
        output, residue_name = process_residue(r)
        if output is not None:
            coordinates.append(output)
            seq.append(residue_name)
    if len(coordinates) == 0:
        return None
    coordinates = np.stack(coordinates)
    seq = ''.join(seq)
    return coordinates, seq

def process_chains(chains, pep=False, prot=False):
    if pep or prot:
        chain_lens = []
        chain_ids = []
        for chain in chains:
            for i, res in enumerate(chain):
                continue
            chain_lens.append(i)
            chain_ids.append(chain.id)
        if chain_lens[0] < chain_lens[1]:
            pep_id = chain_ids[0]
            prot_id = chain_ids[1]
        else:
            pep_id = chain_ids[1]
            prot_id = chain_ids[0]
        if pep and isinstance(pep, str): pep_id == pep
        if prot and isinstance(prot, str): prot_id == prot
    output = []
    chain_ids = []
    for chain in chains:
        if (pep and chain.id != pep_id) or (prot and chain.id != prot_id):
            continue
        out = process_chain(chain)
        if out is not None:
            output.append(out)
            chain_ids.append(chain.id)
    coords = [u[0] for u in output]
    seqs = [u[1] for u in output]
    return coords, seqs, chain_ids

def process_structure(structure, pep=False, prot=False):
    for s in structure: # only one structure
        return process_chains(s, pep, prot)
    return None

# +
def process_pdb(parser, pdb_filename):
    # print(pdb_filename)
    with gzip.open(pdb_filename, "rt") as file_handle:
        structure = parser.get_structure("?", file_handle)
        date = structure.header['deposition_date']
        return process_structure(structure), date
    
def process_pdb_raw(parser, pdb_filename, pep=False, prot=False):
    s = parser.get_structure("?", pdb_filename)
    return process_structure(s, pep, prot)

def read_input_ids(index_file):
    input_ids = []
    with open(os.path.join(index_file), 'r') as f:
        for line in f:
            input_ids += [line.strip()]
    return np.array(input_ids)

def write_dataset(dataset, tar_name, use_shards=False, max_shard_count=10000):
    if use_shards:
        os.makedirs(tar_name, exist_ok=True)
        sink = wds.ShardWriter(f'{tar_name}/shard-%06d.tar',maxcount=max_shard_count)
    else:
        sink = wds.TarWriter(tar_name)
    for index, (batch, pdb_id) in enumerate(dataset):
        if index%1000==0:
            print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
        if len(batch[0]) == 0:
            continue
        sink.write({
            "__key__": "sample%06d" % index,
            "inp.pyd": dict(coords=batch[0], seqs=batch[1], chain_ids=batch[2], pdb_id=pdb_id),
        })
    sink.close()
    
def make_wds(dir_, tar_):
    """
    Args:
        dir_ (str): Directory containing PDB files.
        tar_ (str): Output file path to write WDS to.
    """
    parser = PDBParser()
    root_pdb = dir_
    outputs = []
    for i, pdb_file in tqdm(enumerate(os.listdir(dir_)), total=len(os.listdir(dir_))):
        pdb_file = pdb_file.strip()
        pdb_file = os.path.join(dir_, pdb_file)
        out = process_pdb_raw(parser, pdb_file)
        pdb_id = pdb_file.split('.')[0]
        outputs.append((out, pdb_id))

    dataset = []
    for o, pdb_id in tqdm(outputs):
        if o is None:
            continue
        dataset.append((o, pdb_id))

    write_dataset(dataset, tar_)


def run_rla(pdb_dir, settings, output_dir):
    """
    Args:
        pdb_dir (str): Directory containing PDB files.
        settings (dict): Dictionary of settings. See example json.
        output_dir (str): Output directory to write WDS to.
    
    settings:
        seq_mask: 'peptide' to mask peptide sequence, 'protein' to mask protein sequence
        struct_mask: 'peptide' to mask peptide structure, 'protein' to mask protein structure
        top_k: Num neighbors, probably want to keep at 30 but can experiment
        focus: RLA calculation setting that limits RLA score to interface, almost certainly want to keep True
        remove_far: Removes residues too far from the interface, likely want to keep True
        threshold: Threshold for remove_far calculation, likely want to keep at 1
        weight_dists: Weights RLA score per residue by distance from interface, likely want to keep False
        force_from_back: Define TRUE if masked sequence is always the last chain
        force_from_front: Define TRUE if masked sequence is always the first chain
    """
    # Make WDS
    design_sets = ['wds.file']
    wds_file = os.path.join(output_dir, 'wds.file')
    make_wds(pdb_dir, wds_file)

    ## GENERAL SETUP (CHANGE PATHS AS NEEDED)
    model_dir = "version_0/" 
    dev = 'cuda:0'
    CLIP_MODE = False
    root_path = os.path.join(ROOT, model_dir)
    path = os.path.join(root_path, "checkpoints/checkpoint_best.pt")
    
    data_root = output_dir
    args_path = os.path.join(ROOT, model_dir, [u for u in os.listdir(os.path.join(ROOT, model_dir)) if u.endswith('.pt')][0])

    backwards_compat = {
        'masked_rate': -1,
        'masked_mode': 'MASK',
        'lm_only_text': 1,
        'lm_weight': 1,
        'resid_weight': 1,
        'language_head': False,
        'language_head_type': 'MLP',
        'zip_enabled': False,
        'num_mutations': False,
    }
    hparams = torch.load(args_path)
    args_dict = hparams['args']
    args_dict['data_root'] = data_root
    args_dict['batch_size'] = 1
    args_dict['blacklist_file'] = ''
    args_dict['num_workers'] = 1
    for k in backwards_compat.keys():
        if k not in args_dict:
            args_dict[k] = backwards_compat[k]
    args = SimpleNamespace(**args_dict)

    coordinator_params = data_utils.get_coordinator_params(args.coordinator_hparams)
    coordinator_params['num_positional_embeddings'] = args.gnn_num_pos_embs
    coordinator_params['zero_out_pos_embs']= args.gnn_zero_out_pos_embs
    coordinator_params['clip_mode'] = True

    ## LOAD MODEL (NO CHANGES NEEDED)
    args_dict['arch'] = '/data1/groups/keating_madry/huggingface/esm2_t30_150M_UR50D'
    trained_model = model_utils.load_model(path, args_dict['arch'], dev)
    tokenizer = EsmTokenizer.from_pretrained(args_dict['arch'])

    esm_arch = args_dict['arch']
    esm_model = EsmModel.from_pretrained(args_dict['arch']) 
    esm_model = esm_model.to(dev)
    esm_model.eval()
    esm, alphabet = esmlib.pretrained.esm2_t30_150M_UR50D()
    # esm, alphabet = esmlib.pretrained.esm1v_t33_650M_UR90S_1()
    esm = esm.eval()
    if dev == 'cuda:0':
        esm = esm.cuda()

    trained_model = trained_model.eval()

    args.batch_size=1
    train_path = os.path.join(args.data_root, args.train_wds_path)
    val_path = os.path.join(args.data_root, args.val_wds_path)

    ## PERFORMS RLA CALCULATION (NO CHANGES NEEDED)
    if CLIP_MODE:
        feature_getter = get_text_and_image_features_clip
    else:
        feature_getter = get_text_and_image_features
        
    torch.multiprocessing.set_sharing_strategy('file_system')
    nclash_dict, Fnat_dict, Fnonnat_dict, LRMS_dict, iRMSDbb_dict, irmsdsc_dict, distance_dict, theta_dict, class_dict = {}, {}, {}, {}, {}, {}, {}, {}, {}
    dicts = [nclash_dict, Fnat_dict, Fnonnat_dict, LRMS_dict, iRMSDbb_dict, irmsdsc_dict, distance_dict, theta_dict, class_dict]
    nclash_data, Fnat_data, Fnonnat_data, LRMS_data, iRMSDbb_data, irmsdsc_data, distance_data, theta_data, class_data = {}, {}, {}, {}, {}, {}, {}, {}, {}
    data_dicts = [nclash_data, Fnat_data, Fnonnat_data, LRMS_data, iRMSDbb_data, irmsdsc_data, distance_data, theta_data, class_data]
    args.batch_size = 1
    args.zip_enabled = False
    args.num_mutations = 0
    args.distributed = 0
    plot_scores = []
    plot_weights = []
    plot_pep_mask = []
    plot_indices = []
    plot_X = []
    plot_seq = []
    paired_res_scores = {}
    scores_stats = {'models': [], 'seqs': [], 'rla_scores': []}
    # result_types = ['nclash', 'fnat', 'fnonnat', 'lrmsd', 'irmsdbb', 'irmsdsc', 'distance', 'theta', 'classification']
    for design_set in design_sets:
        args.train_wds_path = f"{design_set}"
        args.val_wds_path = f"{design_set}"
        train_loader, val_loader, train_len, val_len = get_wds_loaders(args, coordinator_params, gpu=None, shuffle_train=False, val_only=True, return_count=False)
        lens = {}
        for i, b in tqdm(enumerate(val_loader), total=val_len):
            lens[b[0]['pdb_id'][0]] = len(b[0]['string_sequence'][0])
        MAX_LEN = max(lens.values())
        
        for i, batch in enumerate(tqdm(val_loader, total=val_len)):
            model = batch[0]['pdb_id'][0]
            pep_seq = batch[0]['string_sequence'][0][:batch[1]['seq_lens'][0][0]]
            chain_lens = torch.zeros(batch[1]['coords'][0].shape[1]).to(device = batch[1]['coords'][0].device)
            chain_lens[batch[1]['seq_lens'][0][0]:] = 1
            chain_lens_mask = torch.ones(batch[1]['coords'][0].shape[1]).unsqueeze(0).to(dtype=torch.bool, device = batch[1]['coords'][0].device)
            batch[1]['chain_lens'] = [chain_lens.unsqueeze(0), chain_lens_mask]
            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    output_dict = feature_getter(trained_model, tokenizer, batch, 
                                                 pdb=None, 
                                                 weight_dists=settings['weight_dists'], 
                                                 seq_mask=settings['seq_mask'], 
                                                 focus=settings['focus'], 
                                                 top_k=settings['top_k'], 
                                                 struct_mask=settings['struct_mask'], 
                                                 remove_far=settings['remove_far'], 
                                                 threshold=settings['threshold'], 
                                                 dev=dev,
                                                 force_from_back=settings['force_from_back'],
                                                 force_from_front=settings['force_from_front']
                                                )
                    score, scores, plot_scores, plot_weights, plot_pep_mask, plot_indices, plot_X, plot_seq = compute_score(batch, output_dict, settings['weight_dists'], MAX_LEN, plot_scores=plot_scores, plot_weights=plot_weights, plot_pep_mask=plot_pep_mask, plot_indices=plot_indices, plot_X=plot_X, plot_seq=plot_seq, is_complex=True)
                    scores_stats['models'].append(model)
                    scores_stats['seqs'].append(pep_seq)
                    paired_res_scores[model] = score


    with open(os.path.join(output_dir, 'seq_mapping.json'), 'w') as f:
        json.dump(scores_stats, f)
    with open(os.path.join(output_dir, 'scores.json'), 'w') as f:
        json.dump(paired_res_scores, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", required=True, help="Directory to input PDBs")
    parser.add_argument("--settings", required=True, help="Path to settings.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    args = parser.parse_args()

    with open(args.settings, 'r') as f:
        settings = json.load(f)

    run_rla(args.pdb_dir, settings, args.output_dir)
