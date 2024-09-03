"""
Implementation of a SMILES dataset from https://github.com/MolecularAI/Reinvent
"""
import math
import random
import torch
import numpy as np
import torch.utils.data as tud

from itertools import permutations
from multiprocessing import cpu_count, get_context, Pool
from functools import partial

def parallel_map(func, iterable, cores = None, chunk=None):
    if cores is None:
        cores = cpu_count() - 1
    if chunk is None:
        chunk = len(iterable) // cores
    with get_context('fork').Pool(cores) as p:
        results =  p.map(func, iterable, chunksize=chunk)
        p.close()
        p.join()
    return results


def dim(a):
    if type(a) == torch.Tensor:
        return list(a.size())
    elif type(a) in [list, np.ndarray]:
        return [len(a)] + dim(a[0])
    else:
        return []


class Dataset(tud.Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, smiles_list, vocabulary, tokenizer):
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._smiles_list = list(smiles_list)

    def __getitem__(self, i):
        smi = self._smiles_list[i]
        tokens = self._tokenizer.tokenize(smi)
        encoded = self._vocabulary.encode(tokens)
        return torch.tensor(encoded, dtype=torch.long)  # pylint: disable=E1102

    def __len__(self):
        return len(self._smiles_list)

    @staticmethod
    def collate_fn(encoded_seqs):
        """Converts a list of encoded sequences into a padded tensor"""
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(
            len(encoded_seqs), max_length, dtype=torch.long
        )  # padded with zeroes
        
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, : seq.size(0)] = seq
        return collated_arr


class CustomDataset(Dataset):
    """ Used for data augmentation"""

    def __init__(self, smiles_list, vocabulary, tokenizer, n_aug=5):
        super().__init__(smiles_list, vocabulary, tokenizer)
        self.n_aug = n_aug
        

    def __getitem__(self, i):
        smi = self._smiles_list[i]
        tokens = self._tokenizer.tokenize(smi, n_aug=self.n_aug)
        #print(f"tokens == {tokens}")
        encoded = self._vocabulary.batch_encode(tokens)
        #print(f"encoded == {encoded}")
        return torch.tensor(encoded, dtype=torch.long)  # pylint: disable=E1102

    @staticmethod
    def collate_fn(encoded_seqs):
        """Converts a list of encoded sequences into a padded tensor"""
        #print(f">>>>>>>>>>>> encoded_seqs size = {dim(encoded_seqs)}")
        
        max_length = max([seq.size(-1) for seq in encoded_seqs])
        collated_arr = []
        
        for i, seq in enumerate(encoded_seqs):
            row = torch.zeros(seq.size(0), max_length, dtype=torch.long)
            row[:,:seq.size(-1)] = seq
            collated_arr.append(row)
        collated_arr = torch.concatenate(collated_arr)
        #print(f">>>>>>>>>>>> collated_arr size = {collated_arr.size()}")
        return collated_arr


def calculate_nlls_from_model(model, smiles, batch_size=128):
    """
    Calculates NLL for a set of SMILES strings.
    :param model: Model object.
    :param smiles: List or iterator with all SMILES strings.
    :return : It returns an iterator with every batch.
    """
    dataset = Dataset(smiles, model.vocabulary, model.tokenizer)
    _dataloader = tud.DataLoader(
        dataset, batch_size=batch_size, collate_fn=Dataset.collate_fn, shuffle=True, 
    )

    def _iterator(dataloader):
        for batch in dataloader:
            nlls = model.likelihood(batch.long())
            yield nlls.data.cpu().numpy()

    return _iterator(_dataloader), len(_dataloader)
