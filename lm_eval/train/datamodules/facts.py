# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
from itertools import chain
from pathlib import Path
import pickle
from typing import Any, List, Optional, Union
import hashlib
import json

from multiprocessing.shared_memory import SharedMemory

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from pytorch_lightning import LightningDataModule

from train.datamodules.fault_tolerant_sampler import RandomFaultTolerantSampler
from train.datamodules.fault_tolerant_sampler import FaultTolerantDistributedSampler
from train.datamodules.datasets.detokenizer import DATASET_TOKENIZATION_REGISTRY
from train.datamodules.language_modeling_hf import SHMArray
from train.utils.utils import get_logger
from train.config import DataModuleConfig

logger = get_logger()


# Inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/tasks/zeroshot_gpt/datasets.py
# Except we don't pad the last block and don't use overlapping eval
# And we return both the input and the target
import math
import numpy as np

import torch



class FactsDataModuleConfig(DataModuleConfig):
    target: str = "train.datamodules.facts.FactsDataModule"
    _pass_as_config: bool = True
    
    dataset_name: str
    tokenizer_name: str
    dataset_config_name: Optional[str]=None
    max_length: int=1024
    max_facts: Optional[int]=None
    cache_dir: Optional[Path]=None
    val_ratio: float=0.0005
    add_eos: bool=True
    val_only: bool=False
    batch_size: int=32
    batch_size_eval: Optional[int]=None
    num_workers: int=1
    shuffle: bool=False
    shuffle_seed: int = 42
    pin_memory: bool=False
    drop_last: bool=False
    fault_tolerant: bool=False
    ddp: bool=False
    use_shmem: bool=False
    force_cache: bool=False


class FactsDataModule(LightningDataModule):
    def __init__(
        self,
        config: FactsDataModuleConfig,
        fast_forward_epochs=None,
        fast_forward_batches=None,
    ):
        super().__init__()       
        if config.batch_size_eval is None:
            config.batch_size_eval = config.batch_size

        self.__dict__.update(config.model_dump())

        # get a cache_dir name that is a hash of the config
        config_string = str(config)
        self._cache_dir_name = hashlib.sha256(config_string.encode()).hexdigest()

        if config.fault_tolerant:
            assert self.shuffle
        if config.ddp:
            assert config.fault_tolerant
        
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if (
            self.fast_forward_epochs is not None
            or self.fast_forward_batches is not None
        ):
            assert config.ddp and config.fault_tolerant

        if config.use_shmem:
            assert config.cache_dir is not None


    def prepare_data(self):
        """This is only run on the main process (unlike setup). 
        So we just download the dataset and then cache the tokenization if we've Got
        a cache_dir.
        """
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self.dataset_name, self.dataset_config_name)
        else:  # Process the dataset and save it
            self.process_dataset(force=self.force_cache)

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        tokenized_dataset, self.tokenizer = self.process_dataset()
        self.vocab_size = len(self.tokenizer)
        # Create all splits
        self.dataset_train, self.dataset_val, self.dataset_test = [
            FactDataset(tokenized_dataset[split], seq_len=self.max_length)
            for split in ["train", "validation", "test"]
        ]

    def process_dataset(self, force: bool=False):
        cache_dir = (
            None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        )
        if cache_dir is not None and not force:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
        # https://github.com/stanford-crfm/mistral/blob/main/src/corpora/auto.py

        if self.max_facts is not None:
            raw_datasets["train"] = raw_datasets["train"].shuffle(
                seed=self.shuffle_seed
            ).select(range(self.max_facts))
        
        # since this is just a fact memorization dataset, we use the same data for 
        # train, val, and test
        raw_datasets["validation"] = raw_datasets["train"]
        raw_datasets["test"] = raw_datasets["train"]

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        def tokenize(row, idx: int):
            # unpack batch
            row = {k: v[0] for k, v in row.items()}
            tokens = []
            mask = []
            template = row["template"].format(subject=row["subject"], object="{object}")

            before, after = template.split("{object}")

            # SE (03/05): We need to rstrip the before and add a space to the object so that 
            # it uses the token with the space ("_bus" vs. "bus").
            # I tested that this works for gemma-2b-it on a simple set of facts
            # NOTE: this is not tested for other tokenizers and it assumes the the object
            # is not the first word in the sentence
            # there are some edge cases where the last token of the object combines with the first 
            # token of the after
            before_tokens = tokenizer.encode(before.rstrip(" "))
            object_tokens = tokenizer.encode(" " + row["object"])[1:]  # remove bos token 
            after_tokens = tokenizer.encode(after)[1:]  + [tokenizer.eos_token_id] # remove bos token and add eos
            tokens = before_tokens + object_tokens + after_tokens
            mask = [0] * len(before_tokens) + [1] * len(object_tokens) + [0] * len(after_tokens)
            indices = np.full(len(tokens), idx)

            return {"mask": mask, "tokens": tokens, "example_idxs": indices}

        tokenized_datasets = raw_datasets.map(
            tokenize,
            batched=True,
            batch_size=1,
            num_proc=max(self.num_workers, 1),
            remove_columns=column_names,
            with_indices=True,
            desc="Running tokenizer on dataset",
        )

        # # explode so that there is one token per row
        # for split in ["train", "validation", "test"]:
        #     tokenized_datasets[split] = tokenized_datasets[split].to_pandas().explode(["tokens", "mask"])



        # if cache_dir is not None:
        #     self._save_to_cache(concat_ids, tokenizer, cache_dir)
        #     if not self.use_shmem:
        #         for name in concat_ids:
        #             Path(cache_dir / f"{name}.bin").unlink()
        return tokenized_datasets, tokenizer

    def _save_to_cache(self, tokenized_datasets, tokenizer, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        
        for k, v in concat_ids.items():
            np.save(cache_dir / f"{k}.npy", v)
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger.info(f"Load from cache at {str(cache_dir)}")
        concat_ids = {
            split: np.load(cache_dir / f"{split}.npy", mmap_mode="r")
            for split in ["train", "validation", "test"]
        }
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return concat_ids, tokenizer

   

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            sampler = (
                FaultTolerantDistributedSampler(self.dataset_train)
                if self.ddp
                else RandomFaultTolerantSampler(self.dataset_train)
            )
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if (
                self.ddp
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": self.fast_forward_batches * self.batch_size,
                    }
                )
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(
        self, *args: Any, **kwargs: Any
    ) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(
        self, dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint["loops"]["fit_loop"][
                "epoch_progress"
            ]["current"]["completed"]
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
                "epoch_loop.batch_progress"
            ]["current"]["completed"]
        # At this point the train loader hasn't been constructed yet



class FactDataset(torch.utils.data.Dataset):

    def __init__(
        self, 
        tokens: Dataset, 
        seq_len: int, 
        drop_last=True
    ):
        """tokens should be a numpy array
        """
        self.seq_len = seq_len
        ntokens = len(tokens)
        if drop_last:
            ntokens = ((ntokens - 1) // seq_len) * seq_len + 1
        self.ntokens = ntokens
        # We're careful not to slice tokens, since it could be a memmap'ed array or H5 dataset,
        # and slicing would load it to memory.
        self.tokens = tokens
        self.total_sequences = math.ceil((self.ntokens - 1) / self.seq_len)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        seq_len = min(self.seq_len, self.ntokens - 1 - start_idx)
        slc = self.tokens[start_idx:(start_idx + seq_len + 1)]
        data = torch.as_tensor(slc["tokens"])
        mask = torch.as_tensor(slc["mask"])
        example_idxs = torch.as_tensor(slc["example_idxs"])
        
        # ignore tokens that aren't the {object} when calculating loss and accuracy
        return data[:-1], data[1:], mask[1:], example_idxs[1:]

