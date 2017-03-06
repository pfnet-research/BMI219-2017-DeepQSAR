import os

from chainer.dataset import download
from chainer import datasets as D
import pandas as pd

from lib.data import kaggle
from lib.data import pubchem


def _filter_row(df):
    cond = ((df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active') |
            (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Inactive'))
    df_new = df[cond]
    df_new.name = df.name
    return df_new


aids = [1851, 1915, 2358, 463213, 463215, 488912, 488915,
        488917, 488918, 492992, 504607, 624504,
        651739, 651744, 652065]


def _creator(cache_path, debug):
    assays = pubchem.get_assay(aids[0], debug)
    assays.extend(pubchem.get_assay(aids[1], debug))
    if not debug:
        for aid in aids[2:]:
            assays.extend(pubchem.get_assay(aid, debug))

    assays = [_filter_row(a) for a in assays]

    # Retrieve smiles
    sids = pubchem.get_sid(assays)
    smiles = pubchem.get_smiles(sids)

    # Save assays
    with pd.HDFStore(cache_path) as store:
        for a in assays:
            store[a.name] = a

    # Save smiles
    with pd.HDFStore(cache_path) as store:
        store['smiles'] = smiles

    return kaggle.create_dataset(assays, smiles, kaggle.smiles2fp)


def get_kaggle(debug=False):
    root = 'pfnet/chainer/pubchem'
    cache_root = download.get_dataset_directory(root)
    fname = 'pubchem_debug.h5' if debug else 'pubchem.h5'
    cache_path = os.path.join(cache_root, fname)

    def creator(path):
        return _creator(path, debug)

    dataset = download.cache_or_load_file(cache_path, creator, kaggle.loader)
    N = len(dataset)
    return D.split_dataset_random(dataset, int(N * 0.8))


if __name__ == '__main__':
    get_kaggle(True)
