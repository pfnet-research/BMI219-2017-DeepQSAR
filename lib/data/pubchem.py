from __future__ import print_function
import json
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem
import requests
import six


def _get_assay(aid, skiprows, usecols, test):
    url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
           'assay/aid/%d/sids/JSON?list_return=listkey' % aid)
    r = requests.get(url)
    j = json.loads(r.text)
    size = j['IdentifierList']['Size']
    listkey = j['IdentifierList']['ListKey']

    if test:
        size = min(100, size)

    ret = []
    for idx in range(0, size, 1000):
        header = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
                  'assay/aid/%d/CSV' % aid)
        parameters = ('?sid=listkey&listkey=%s&listkey_start=%d'
                      '&listkey_count=%d' % (listkey, idx, 1000))
        url = header + parameters
        r = requests.get(url)
        t = pd.read_csv(six.moves.cStringIO(r.text), header=0,
                        skiprows=skiprows, usecols=usecols)
        ret.append(t)
    ret = pd.concat(ret, ignore_index=False)
    ret.name = 'assay_%d' % aid
    return ret,


def _get_assay_1851(test):
    if test:
        assay_num = 2
        cols = 1, 11, 38
    else:
        assay_num = 5
        cols = 1, 11, 38, 65, 92, 119

    df = _get_assay(1851, range(1, 9), cols, test)[0]
    ret = []
    for i in range(assay_num):
        outcome_key = 'Activity Outcome'
        outcome_key += '' if i == 0 else '.' + str(i)
        df_one = pd.DataFrame(
            {'PUBCHEM_SID': df['PUBCHEM_SID'],
             'PUBCHEM_ACTIVITY_OUTCOME': df[outcome_key],
             })
        df_one.name = 'assay_1851_%d' % i
        ret.append(df_one)
    return ret


def get_assay(aid, test=False):
    if aid == 1851:
        return _get_assay_1851(test)
    elif aid == 1915:
        return _get_assay(1915, range(1, 6), (1, 3), test)
    else:
        return _get_assay(aid, range(1, 5), (1, 3), test)


def get_sid(assays):
    sids = []
    for a in assays:
        sids.append(a['PUBCHEM_SID'])
    sids = tuple(sorted(set(pd.concat(sids))))
    sids = pd.Series(sids, dtype=np.int32)
    return sids


def _grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return six.moves.zip_longest(*args, fillvalue=fillvalue)


def get_smiles(sids):
    sids_new = []
    smiles = []
    for s in _grouper(sids, 100):
        s = filter(lambda s: s is not None, s)
        url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/'
               'substance/sid/%s/SDF' % ','.join(map(str, s)))
        r = requests.get(url)

        with tempfile.NamedTemporaryFile() as f:
            f.write(r.text)
            mol_supplier = Chem.SDMolSupplier(f.name)

        for mol in mol_supplier:
            try:
                sid = int(mol.GetProp('PUBCHEM_SUBSTANCE_ID'))
                smiles_ = Chem.MolToSmiles(mol)
                sids_new.append(sid)
                smiles.append(smiles_)
            except Exception as e:
                print(e)

    return pd.DataFrame({'PUBCHEM_SID': sids_new, 'SMILES': smiles})


def concat_assays(assays):
    assays_new = []
    for a in assays:
        # Rename the outcome column to
        # the original dataframe name
        c = list(a.columns)
        idx = c.index('PUBCHEM_ACTIVITY_OUTCOME')
        c[idx] = a.name
        a.columns = c

        assays_new.append(a)

    def merge(left, right):
        return pd.merge(left, right, on='PUBCHEM_SID', how='outer')

    assays = six.moves.reduce(merge, assays)
    return assays
