import os

from chainer.dataset import download
from chainer import datasets as D
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as desc
import shutil

import pubchem


task_names = None
data_url = 'https://www.dropbox.com/s/g25vyeralmba4d0/pubchem.h5?raw=1'


def smiles2fp(smiles, radius=6, n_bits=4096):
    mol = Chem.MolFromSmiles(smiles)
    try:
        return desc.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    except Exception as e:
        print(e)
        return -1


def create_dataset(assays, smiles, featurizer):
    # Merge assay results
    assays = pubchem.concat_assays(assays)
    df = pd.merge(assays, smiles, on='PUBCHEM_SID', how='inner')

    # Convert smiles to fingerprint and drop substances
    # that cannot be converted to fingerprint.
    print('Creating feature vectors from SMILEs...')
    df['FINGERPRINT'] = df['SMILES'].apply(featurizer)
    df = df[df['FINGERPRINT'] != -1]
    fps = np.array(list(df['FINGERPRINT'].values), dtype=np.float32)

    # Convert outcome to binary value
    assays = df.drop(['PUBCHEM_SID', 'SMILES', 'FINGERPRINT'], axis=1).values
    assays[assays == 'Active'] = 0
    assays[assays == 'Inactive'] = 1
    assays[(assays != 0) & (assays != 1)] = -1
    assays = assays.astype(np.int32)

    assert len(fps) == len(assays)
    return D.TupleDataset(fps, assays)


def _load(path):
    with pd.HDFStore(path) as store:
        task_names = filter(lambda name: name.startswith('/assay'),
                            store.keys())
        assays = []
        for t in task_names:
            a = store[t]
            a.name = t
            assays.append(a)
        smiles = store['smiles']
    return assays, smiles, task_names


def creator(cache_path):
    data_path = download.cached_download(data_url)
    shutil.copy(data_path, cache_path)

    global task_names
    assays, smiles, task_names = _load(data_path)
    return create_dataset(assays, smiles, smiles2fp)


def loader(cache_path):
    global task_names
    assays, smiles, task_names = _load(cache_path)
    return create_dataset(assays, smiles, smiles2fp)


def get_kaggle():
    root = 'pfnet/chainer/pubchem'
    cache_root = download.get_dataset_directory(root)
    fname = 'pubchem.h5'
    cache_path = os.path.join(cache_root, fname)

    dataset = download.cache_or_load_file(cache_path, creator, loader)
    N = len(dataset)
    return D.split_dataset_random(dataset, int(N * 0.8))
