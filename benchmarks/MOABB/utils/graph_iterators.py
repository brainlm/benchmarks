"""
Graph Data Iterators for MOABB Datasets and Paradigms

This module provides various data iterators tailored for MOABB datasets and paradigms which
require the electrode position and adjacency data.

Different training strategies have been implemented as distinct data iterators, including:
- Leave-One-Session-Out
- Leave-One-Subject-Out

Authors
------
Drew Wagner, 2024
"""

import abc
import random
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
import torch
from mne import EpochsArray
from mne.channels import find_ch_adjacency
from moabb import paradigms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ChainDataset, IterableDataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from mne.utils.config import set_config, get_config


def rename_subjects(dataset, subjects):
    if isinstance(subjects, callable):
        dataset.subject_list = subjects(dataset.subject_list)
    elif isinstance(subjects, dict):
        dataset.subject_list = [subjects[sub] for sub in dataset.subject_list]
    elif isinstance(dataset, (list, tuple)):
        dataset.subject_list = subjects

    return dataset


def _make_paradigm(
    dataset, *, fmax, resample=128, fmin=0, tmin=0, tmax=None, **kwargs
):
    if tmax is None:
        tmax = dataset.interval[1] - dataset.interval[0]

    Paradigm = dict(
        imagery=paradigms.MotorImagery,
        p300=paradigms.P300,
        ssvep=paradigms.SSVEP,
    )[dataset.paradigm]

    if dataset.paradigm == "p300":
        kwargs.pop("events")

    return Paradigm(
        resample=resample, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, **kwargs
    )


class PositionNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        x.pos = x.pos + torch.randn_like(x.pos) * self.sigma
        return x


class NodeDrop(nn.Module):
    def __init__(self, choose_k):
        super().__init__()
        self.k = choose_k

    def forward(self, x):
        del x.edge_index
        x = x.to_data_list()
        for d in x:
            mask = torch.randperm(d.x.shape[0], device=d.x.device)[: self.k]
            d.x = d.x[mask, :]
            d.pos = d.pos[mask, :]

        return Batch.from_data_list(x)


class AsGeometricData(IterableDataset):
    X: EpochsArray
    Y: np.ndarray
    metadata: pd.DataFrame
    pos: np.ndarray
    adjacency_mtx: np.ndarray
    edge_list: np.ndarray

    def __init__(
        self,
        dataset,
        target_subjects=None,
        **paradigm_kwargs,
    ):
        self.dataset = dataset
        self.paradigm = _make_paradigm(dataset, **paradigm_kwargs)
        self.subjects = target_subjects

        self.is_loaded = False

    def load(self):
        self.X, self.Y, self.metadata = self.paradigm.get_data(
            self.dataset,
            subjects=self.subjects,
            return_epochs=True,
            cache_config=dict(save_epochs=True, use=True),
        )

        # NOTE: Assumes a standard headsphere with offset 0,0,40mm and radius 90mm
        offset = torch.tensor([[0, 0, 0.04]])  # meters
        radius = 0.09  # meters

        self.pos = (
            torch.tensor(
                list(
                    self.X.info.get_montage().get_positions()["ch_pos"].values()
                )
            )
            .sub_(offset)
            .div_(radius)
            .float()
            .contiguous()
        )
        adjacency_mtx, _ = find_ch_adjacency(self.X.info, ch_type="eeg")
        self.adjacency_mtx = scipy.sparse.csr_matrix.toarray(
            adjacency_mtx
        )  # from sparse mtx to ndarray
        self.edge_list = np.vstack(np.nonzero(adjacency_mtx))
        self.edge_list = torch.from_numpy(self.edge_list).float().contiguous()

        self.is_loaded = True

    def __iter__(self):
        if not self.is_loaded:
            self.load()

        for x, y, (_, m) in zip(self.X, self.Y, self.metadata.iterrows()):
            yield Data(
                x=torch.from_numpy(x).float(),
                y=self.dataset.event_id[y] - 1,
                edge_index=self.edge_list,
                pos=self.pos,
                **m,
            )


class BaseGraphData(abc.ABC):
    train: DataLoader
    valid: DataLoader
    test: DataLoader

    def __init__(
        self,
        datasets,
        target_subjects=None,
        target_sessions=None,
        valid_ratio=0.2,
        **paradigm_kwargs,
    ):
        if not isinstance(datasets, Iterable):
            datasets = [datasets]

        self.datasets = datasets
        if not isinstance(target_subjects, (list, tuple)):
            target_subjects = [target_subjects]
        if not isinstance(target_sessions, (list, tuple)):
            target_sessions = [target_sessions]
        self.target_subjects = list(map(int, target_subjects))
        self.target_sessions = list(map(int, target_sessions))
        self.target_session_names = None
        self.data = ChainDataset(self.make_geometric_data(paradigm_kwargs))
        self.valid_ratio = valid_ratio

    @abc.abstractmethod
    def make_geometric_data(self, paradigm_kwargs): ...

    @abc.abstractmethod
    def split_test_data(self, data): ...

    def data_statistics(self, data):
        min_C = np.Inf
        max_C = 0
        min_T = np.Inf
        max_T = 0

        for d in data:
            C, T, *_ = d.x.shape
            min_C = min(min_C, C)
            max_C = max(max_C, C)
            min_T = min(min_T, T)
            max_T = max(max_T, T)

        return min_C, max_C, min_T, max_T

    def pad_data(self, data, max_T):
        for d in data:
            T = d.x.shape[1]
            if T < max_T:
                d.x = F.pad(d.x, (0, max_T - T))
        return data

    def prepare(
        self,
        data_folder=None,
        cached_data_folder=None,
        batch_size=1,
        exclude_keys=None,
    ):
        if data_folder is not None:
            mne_cfg = get_config()
            for a in mne_cfg.keys():
                if (
                    mne_cfg[a] != data_folder
                ):  # reducing writes on mne cfg file to avoid conflicts in parallel trainings
                    set_config(a, data_folder)
            if cached_data_folder is None:
                cached_data_folder = data_folder

        data = list(self.data)
        min_C, max_C, min_T, max_T = self.data_statistics(data)
        data = self.pad_data(data, max_T)
        train_valid_data, test_data = self.split_test_data(data)
        assert (
            len(train_valid_data) > 0
        ), "Invalid Split: No Training & Validation Data"
        assert len(test_data) > 0, "Invalid Split: No Test Data"

        random.shuffle(train_valid_data)
        train_valid_data = Batch.from_data_list(train_valid_data)

        valid_idxs = []
        for c in train_valid_data.y.unique():
            c_idxs = np.nonzero(train_valid_data.y == c)
            idx = np.linspace(
                0,
                c_idxs.shape[0] - 1,
                round(self.valid_ratio * c_idxs.shape[0]),
            ).astype(int)
            valid_idxs.extend(c_idxs[idx])

        valid_idxs = np.array(valid_idxs)
        train_idxs = np.setdiff1d(
            np.arange(len(train_valid_data.y)), valid_idxs
        )

        train = train_valid_data.index_select(train_idxs)
        valid = train_valid_data.index_select(valid_idxs)

        if exclude_keys is None:
            exclude_keys = []
        exclude_keys += ["subject", "session", "run"]

        self.train = DataLoader(
            train,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
            exclude_keys=exclude_keys,
        )
        self.valid = DataLoader(
            valid,
            batch_size=batch_size,
            pin_memory=True,
            exclude_keys=exclude_keys,
        )
        self.test = DataLoader(
            test_data,
            batch_size=batch_size,
            pin_memory=True,
            exclude_keys=exclude_keys,
        )

        return {
            "train": self.train,
            "valid": self.valid,
            "test": self.test,
            "T": max_T,
            "C": max_C,
            "min_T": min_T,
            "min_C": min_C,
        }


class LeaveOneSessionOut(BaseGraphData):
    def make_geometric_data(self, paradigm_kwargs):
        return [
            AsGeometricData(
                dataset,
                # Pass target subject for IO efficiency
                target_subjects=list(
                    set(self.target_subjects).intersection(dataset.subject_list)
                ),
                **paradigm_kwargs,
            )
            for dataset in self.datasets
            if set(self.target_subjects).intersection(dataset.subject_list)
        ]

    def split_test_data(self, data):
        other, test = [], []
        sessions = set()
        target_sessions = set()
        for d in data:
            sessions.add(d.session)
            sess_idx = sorted(sessions).index(d.session)
            if sess_idx in self.target_sessions:
                target_sessions.add(d.session)
                test.append(d)
            else:
                other.append(d)

        self.target_session_names = target_sessions
        return other, test


class LeaveOneSubjectOut(BaseGraphData):
    def make_geometric_data(self, paradigm_kwargs):
        return [
            AsGeometricData(
                dataset,
                **paradigm_kwargs,
            )
            for dataset in self.datasets
        ]

    def split_test_data(self, data):
        other, test = [], []
        for d in data:
            if d.subject in self.target_subjects:
                test.append(d)
            else:
                other.append(d)
        return other, test
