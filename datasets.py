import torch
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset


class PretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, 'r')
        keys = [
            'input_ids',
            'input_mask',
            'segment_ids',
            'masked_lm_positions',
            'masked_lm_ids',
            'next_sentence_labels',
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        print(f"Loaded {len(self.inputs[0]):d} samples from file: {input_file}")
        f.close()

    def __len__(self):
        return len(self.inputs[0])
    
    def __getitem__(self, index):
        [
            input_ids,
            input_mask,
            segment_ids,
            masked_lm_positions,
            masked_lm_ids,
            next_sentence_labels,
        ] = [
            torch.from_numpy(input[index].astype(np.int64))
            if indice < 5 else torch.from_numpy(np.asarray(input[index].astype(np.int64)))
            for indice, input in enumerate(self.inputs)
        ]
        masked_lm_labels = torch.zeros(input_ids.shape, dtype=torch.long) - 100
        index = self.max_pred_length
        masked_token_count = torch.count_nonzero(masked_lm_positions)
        if masked_token_count != 0:
            index = masked_token_count
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        return [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ]


def get_pretraining_datafiles(dirname):
    datafiles = [os.path.join(dirname, f) for f in os.listdir(dirname) \
        if os.path.isfile(os.path.join(dirname, f)) and 'pretrain-part' in f]
    datafiles.sort()
    return datafiles


def get_pretraining_dataloader(input_file, batch_size, max_pred_length):
    dataset = PretrainingDataset(input_file, max_pred_length)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return loader


if __name__ == '__main__':
    from data.tokenization import FullTokenizer
    tokenizer = FullTokenizer('/data/wiki/bert-base-uncased-vocab.txt')
    datafiles = get_pretraining_datafiles('/data/wiki/results/hdf5')
    dataloader = get_pretraining_dataloader(datafiles[0], 10, 32)

    for data in dataloader:
        [
            input_ids,
            segment_ids,
            input_mask,
            masked_lm_labels,
            next_sentence_labels,
        ] = data
        idss = input_ids.tolist()
        for ids in idss:
            tokens = tokenizer.convert_ids_to_tokens(ids)
            print(" ".join(tokens))
        # print(data[0])
        raise
