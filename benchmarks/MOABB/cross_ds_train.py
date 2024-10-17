import json
import random

import math
import pathlib as pl
import warnings
from contextlib import redirect_stdout
from functools import cached_property
from pprint import pformat
from typing import Any, Optional, Sequence, Dict, List

import numpy as np
import pandas as pd
import torch
from moabb.datasets import BNCI2014_001, Cho2017, PhysionetMI
from moabb.datasets.base import BaseDataset
from moabb.paradigms import MotorImagery
from moabb.paradigms.base import BaseParadigm
from models.SpatialEEGNet import SpatialEEGNet, SpatialFocus
from orion.client import build_experiment, ExperimentClient
from sklearn import metrics
from speechbrain.nnet.losses import nll_loss
from speechbrain.processing.signal_processing import mean_std_norm
from torch.utils.data import ConcatDataset, Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
import logging
from torch.utils.tensorboard import (
    SummaryWriter,
)  # Added import for TensorBoard

# ==========================
# Configuration and Constants
# ==========================
SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESSING_PARAMS = {
    "fmin": 0.1,
    "fmax": 50.0,
    "resample": 128,
}

CACHE_CONFIG = {"save_epochs": True, "use": True}

PAD_TIME = 513
N_CLASSES = 3
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4  # Effective Batchsize = 32
MAX_TRIALS = 100
SEARCH_SPACE = {
    "learning_rate": "choices([0.01, 0.005, 0.001, 0.0005, 0.0001])",
    "number_of_epochs": "fidelity(1, 120, base=4)",
    "spatial_focus_projection_dim": "uniform(4, 44, discrete=True)",
    "spatial_focus_temperature": "uniform(0.01, 1.0, precision=4)",
    "cnn_temporal_kernels": "uniform(32, 72, discrete=True)",
    "cnn_temporal_kernelsize": "uniform(24, 62, discrete=True)",
    "cnn_spatial_depth_multiplier": "uniform(1, 4, discrete=True)",
    "cnn_septemporal_point_kernels_ratio_": "uniform(0, 8, discrete=True)",
    "cnn_septemporal_kernelsize_": "uniform(3, 24, discrete=True)",
    "cnn_septemporal_pool": "uniform(1, 8, discrete=True)",
    "dropout": "uniform(0.0, 0.5)",
}
EXPERIMENT_NAME = "spatial-eegnet-beetl-MI"
WORKING_DIR = f"results/hopt/{EXPERIMENT_NAME}"
pl.Path(WORKING_DIR).mkdir(exist_ok=True, parents=True)


# ==========================
# Logging Configuration
# ==========================

# Configure logging to output to both console and a file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{WORKING_DIR}/training.log"),
        logging.StreamHandler(),
    ],
)
logging.getLogger("moabb").setLevel(logging.ERROR)
logging.getLogger("mne").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logger.info(f"Using device: {DEVICE}")

# ==========================
# Dataset Preparation
# ==========================


class TorchMOABBDataset(Dataset):
    def __init__(
        self,
        dataset: BaseDataset,
        paradigm: BaseParadigm,
        subjects: Optional[Sequence[int]] = None,
        map_labels: Optional[Dict[str, int]] = None,
        cache_config: Optional[Dict] = None,
        head_sphere: tuple = ((0, 0, 0.04), 0.09),  # offset, radius
        pad_time: Optional[int] = None,
    ):
        self.dataset = dataset
        self.paradigm = paradigm
        self.subjects = subjects or dataset.subject_list
        self.map_labels = map_labels
        self.cache_config = cache_config
        self.head_sphere = head_sphere
        self.pad_time = pad_time

    def __len__(self):
        return len(self._data[0])

    def __getitem__(self, index):
        X, y, metadata, ch_positions = self._data
        meta = metadata.iloc[index]  # Corrected to use the current index

        x = X[index]
        if self.pad_time:
            assert (
                self.pad_time >= x.shape[1]
            ), "Expected T to be less than pad_time"
            x = torch.nn.functional.pad(
                x, (0, self.pad_time - x.shape[1], 0, 0)
            )

        return Data(
            x=x.to(DEVICE),
            y=y[index].to(DEVICE),
            pos=ch_positions.to(DEVICE),
            subject=meta["subject"],
            session=meta["session"],
            run=meta["run"],
        )

    @cached_property
    def _data(self):
        from io import StringIO

        # Suppress all the garbage output from MOABB / MNE
        with warnings.catch_warnings(), redirect_stdout(StringIO()):
            warnings.simplefilter("ignore")
            X, y, metadata = self.paradigm.get_data(
                self.dataset,
                subjects=self.subjects,
                return_epochs=True,
                cache_config=self.cache_config,
            )

        # Prepare channel positions
        offset = torch.tensor(self.head_sphere[0])
        radius = self.head_sphere[1]
        ch_positions = (
            torch.from_numpy(
                np.array(
                    list(
                        X.info.get_montage().get_positions()["ch_pos"].values()
                    )
                )
            )
            .sub_(offset)
            .div_(radius)
            .float()
            .contiguous()
        )

        # Prepare features
        X = torch.from_numpy(X.get_data()).float()
        X = mean_std_norm(X, dims=(1, 2))

        # Prepare labels
        y = pd.Series(y)
        if self.map_labels:
            y = y.replace(self.map_labels)
        else:
            y = y.replace(self.dataset.event_id) - 1
        # Ensures that all labels were converted:
        y = pd.to_numeric(y, errors="raise", downcast="unsigned")
        y = torch.from_numpy(y.values)

        # Convert session and run names to categorical indices
        metadata["session"], self._session_names = metadata[
            "session"
        ].factorize(sort=True)
        metadata["run"], self._run_names = metadata["run"].factorize(sort=True)

        return X, y, metadata, ch_positions


def prepare_datasets() -> ConcatDataset:
    bnci = TorchMOABBDataset(
        dataset=BNCI2014_001(),
        paradigm=MotorImagery(**PREPROCESSING_PARAMS),
        cache_config=CACHE_CONFIG,
        map_labels=dict(left_hand=0, right_hand=1, feet=2, tongue=2),
        pad_time=PAD_TIME,
    )
    cho = TorchMOABBDataset(
        dataset=Cho2017(),
        paradigm=MotorImagery(**PREPROCESSING_PARAMS),
        cache_config=CACHE_CONFIG,
        map_labels=dict(left_hand=0, right_hand=1),
        pad_time=PAD_TIME,
    )

    return ConcatDataset([bnci, cho])


def train_valid_split(dataset: ConcatDataset, valid_ratio: float = 0.2):
    N = len(dataset)
    N_valid = int(N * valid_ratio)
    return random_split(dataset, [N - N_valid, N_valid])


# ==========================
# Model Definition
# ==========================


def get_model(
    T: int,
    n_classes: int,
    spatial_focus_projection_dim: int,
    spatial_focus_temperature: float,
    cnn_temporal_kernels: int,
    cnn_temporal_kernelsize: int,
    cnn_spatial_depth_multiplier: int,
    cnn_septemporal_point_kernels_ratio_: float,
    cnn_septemporal_kernelsize_: int,
    cnn_septemporal_pool: int,
    dropout: float,
) -> SpatialEEGNet:
    activation_type = "elu"
    cnn_spatial_max_norm = 1
    cnn_spatial_pool = 4
    cnn_spatial_depth_multiplier = 4
    cnn_septemporal_depth_multiplier = 1

    cnn_septemporal_point_kernels_ratio = (
        cnn_septemporal_point_kernels_ratio_ / 4
    )
    # Number of temporal filters in the separable temporal conv. layer
    cnn_septemporal_point_kernels_ = (
        cnn_temporal_kernels
        * cnn_spatial_depth_multiplier
        * cnn_septemporal_depth_multiplier
    )
    cnn_septemporal_point_kernels = math.ceil(
        cnn_septemporal_point_kernels_ratio * cnn_septemporal_point_kernels_ + 1
    )
    max_cnn_spatial_pool = 4
    cnn_septemporal_kernelsize = round(
        cnn_septemporal_kernelsize_ * max_cnn_spatial_pool / cnn_spatial_pool
    )
    cnn_pool_type = "avg"
    dense_max_norm = 0.25

    model = SpatialEEGNet(
        T=T,
        C=spatial_focus_projection_dim,
        cnn_temporal_kernels=cnn_temporal_kernels,
        cnn_temporal_kernelsize=[cnn_temporal_kernelsize, 1],
        cnn_spatial_depth_multiplier=cnn_spatial_depth_multiplier,
        cnn_spatial_max_norm=cnn_spatial_max_norm,
        cnn_spatial_pool=[cnn_spatial_pool, 1],
        cnn_septemporal_depth_multiplier=cnn_septemporal_depth_multiplier,
        cnn_septemporal_point_kernels=cnn_septemporal_point_kernels,
        cnn_septemporal_kernelsize=[cnn_septemporal_kernelsize, 1],
        cnn_septemporal_pool=[cnn_septemporal_pool, 1],
        cnn_pool_type=cnn_pool_type,
        activation_type=activation_type,
        spatial_focus=SpatialFocus(
            projection_dim=spatial_focus_projection_dim,
            position_dim=3,
            tau=spatial_focus_temperature,
        ),
        dense_max_norm=dense_max_norm,
        dropout=dropout,
        dense_n_neurons=n_classes,
    )
    return model.to(DEVICE)


# ==========================
# Training and Evaluation
# ==========================


def compute_class_weights(
    train_dataset: Dataset, n_classes: int
) -> torch.Tensor:
    train_labels = [
        train_dataset[i].y.item() for i in range(len(train_dataset))
    ]
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    # Ensure all classes are present
    if len(class_counts) < n_classes:
        for cls in range(n_classes):
            if cls not in class_counts:
                class_counts.loc[cls] = 0
    class_counts = class_counts.sort_index()
    class_weights = class_counts / class_counts.max()
    class_weights = torch.from_numpy(class_weights.values).float().to(DEVICE)
    return class_weights


def evaluate_hparams(
    checkpoint: pl.Path,
    batch_size: int,
    learning_rate: float,
    number_of_epochs: float,  # Accept as float
    spatial_focus_projection_dim: int,
    spatial_focus_temperature: float,
    cnn_temporal_kernels: int,
    cnn_temporal_kernelsize: int,
    cnn_spatial_depth_multiplier: int,
    cnn_septemporal_point_kernels_ratio_: float,
    cnn_septemporal_kernelsize_: int,
    cnn_septemporal_pool: int,
    dropout: float,
) -> List[Dict[str, Any]]:
    set_seed(SEED)
    # Cast to integer
    number_of_epochs = int(round(number_of_epochs))
    logger.info(f"Number of Epochs (after casting): {number_of_epochs}")

    # Prepare DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize Model
    model = get_model(
        T=PAD_TIME,
        n_classes=N_CLASSES,
        spatial_focus_projection_dim=spatial_focus_projection_dim,
        spatial_focus_temperature=spatial_focus_temperature,
        cnn_temporal_kernels=cnn_temporal_kernels,
        cnn_temporal_kernelsize=cnn_temporal_kernelsize,
        cnn_spatial_depth_multiplier=cnn_spatial_depth_multiplier,
        cnn_septemporal_point_kernels_ratio_=cnn_septemporal_point_kernels_ratio_,
        cnn_septemporal_kernelsize_=cnn_septemporal_kernelsize_,
        cnn_septemporal_pool=cnn_septemporal_pool,
        dropout=dropout,
    )

    # Define Optimizer and Schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=number_of_epochs
    )
    swa_scheduler = SWALR(
        optimizer,
        anneal_strategy="linear",
        anneal_epochs=1 + int(0.75 * number_of_epochs),
        swa_lr=0.05,
    )
    swa_model = AveragedModel(model)
    swa_start = 1 + int(0.75 * number_of_epochs)
    logger.info(f"SWA will start after epoch {swa_start}")

    # Initialize TensorBoard SummaryWriter
    tensorboard_log_dir = checkpoint / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    logger.info(
        f"Initialized TensorBoard SummaryWriter at {tensorboard_log_dir}"
    )

    # Optionally, log hyperparameters
    writer.add_hparams(
        {
            "learning_rate": learning_rate,
            "number_of_epochs": number_of_epochs,
            "spatial_focus_projection_dim": spatial_focus_projection_dim,
            "spatial_focus_temperature": spatial_focus_temperature,
            "cnn_temporal_kernels": cnn_temporal_kernels,
            "cnn_temporal_kernelsize": cnn_temporal_kernelsize,
            "cnn_spatial_depth_multiplier": cnn_spatial_depth_multiplier,
            "cnn_septemporal_point_kernels_ratio_": cnn_septemporal_point_kernels_ratio_,
            "cnn_septemporal_kernelsize_": cnn_septemporal_kernelsize_,
            "cnn_septemporal_pool": cnn_septemporal_pool,
            "dropout": dropout,
        },
        {},
    )

    # Training Loop
    model.train()
    logger.info("Starting training...")
    for epoch in range(1, number_of_epochs + 1):
        running_loss = 0.0
        num_batches = 0

        optimizer.zero_grad()
        for idx, batch in enumerate(train_loader):
            # Forward pass
            output = model(batch)
            loss = nll_loss(output, batch.y, weight=class_weights)
            loss.backward()

            running_loss += loss.item()
            num_batches += 1
            if idx % GRADIENT_ACCUMULATION == GRADIENT_ACCUMULATION - 1:
                # Backward pass and optimization
                optimizer.step()
                optimizer.zero_grad()

        # Calculate average loss for the epoch
        avg_loss = running_loss / num_batches
        logger.info(
            f"Epoch {epoch}/{number_of_epochs} - Training Loss: {avg_loss:.4f}"
        )

        # Log training loss to TensorBoard
        writer.add_scalar("Train/Loss", avg_loss, epoch)

        # Update schedulers
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            logger.debug(
                "Updated SWA model parameters and stepped SWA scheduler."
            )
        else:
            scheduler.step()
            logger.debug("Stepped regular scheduler.")

    # SWA Update with training data
    logger.info("Starting SWA updating with training data...")
    swa_model.eval()
    with torch.no_grad():
        for batch in train_loader:
            swa_model(batch)

    # Save the SWA model
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_to_save = (
        swa_model.module if hasattr(swa_model, "module") else swa_model
    )
    model_save_path = checkpoint / "model.pth"
    torch.save(model_to_save.state_dict(), model_save_path)
    logger.info(f"Saved SWA model to {model_save_path}")

    # Evaluation
    logger.info("Starting evaluation on validation set...")
    swa_model.eval()
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch in valid_loader:
            output = swa_model(batch)
            y_pred = torch.argmax(output, dim=-1).cpu().numpy()
            y_true = batch.y.cpu().numpy()

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

    # Compute Metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = metrics.balanced_accuracy_score(y_true_all, y_pred_all)
    f1 = metrics.f1_score(y_true_all, y_pred_all, average="macro")
    cm = metrics.confusion_matrix(y_true_all, y_pred_all).tolist()

    logger.info(f"Validation Balanced Accuracy: {acc:.4f}")
    logger.info(f"Validation F1 Score (Macro): {f1:.4f}")
    logger.info(f"Validation Confusion Matrix: {cm}")

    # Log validation metrics to TensorBoard
    writer.add_scalar("Validation/Balanced_Accuracy", acc)
    writer.add_scalar("Validation/F1_Score_Macro", f1)
    # Logging confusion matrix as a figure
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        metrics.confusion_matrix(y_true_all, y_pred_all),
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_figure = plt.gcf()
    writer.add_figure("Validation/Confusion_Matrix", cm_figure)
    plt.close()

    # Close the TensorBoard writer
    writer.close()
    logger.info(f"Closed TensorBoard SummaryWriter at {tensorboard_log_dir}")

    return [
        {"name": "err", "type": "objective", "value": 1 - acc},
        {"name": "acc", "type": "statistic", "value": acc},
        {"name": "f1", "type": "statistic", "value": f1},
        {"name": "cm", "type": "statistic", "value": cm},
    ]


# ==========================
# Hyperparameter Optimization
# ==========================


def setup_hyperparameter_optimization() -> ExperimentClient:
    experiment = build_experiment(
        name=EXPERIMENT_NAME,
        algorithm=dict(
            bohb=dict(seed=SEED),
        ),
        space=SEARCH_SPACE,
        max_trials=MAX_TRIALS,
        working_dir=WORKING_DIR,
    )
    experiment.algorithm.seed_rng(SEED)

    logger.info("Hyperparameter optimization experiment initialized.")
    return experiment


def run_hyperparameter_search(experiment: ExperimentClient):
    trial_idx = 0
    while not experiment.is_done:
        trial_idx += 1
        trial = experiment.suggest()
        if trial is None and experiment.is_done:
            logger.info("All trials have been completed.")
            break

        checkpoint = pl.Path(experiment.working_dir) / str(trial.hash_params)
        checkpoint.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trial {trial_idx}/{MAX_TRIALS}")
        logger.info("Hyperparameters:")
        pprint_log(trial.params)

        try:
            stats = evaluate_hparams(
                checkpoint=checkpoint, batch_size=BATCH_SIZE, **trial.params
            )
            if stats:
                logger.info("Validation Results:")
                pprint_log(stats)
                # Log trial parameters and results to a JSON file
                with open(checkpoint / "results.json", "w") as f:
                    json.dump(
                        {"params": trial.params, "metrics": stats}, f, indent=4
                    )
                experiment.observe(trial, stats)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"Error during trial {trial_idx + 1}: {e}")
            experiment.observe(
                trial,
                [
                    dict(name="err", type="objective", value=1e10)
                ],  # Report bad trial
            )

# ==========================
# Main Execution Flow
# ==========================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pprint_log(obj, level=logging.INFO):
    for line in pformat(obj).split("\n"):
        logger.log(level, line)


if __name__ == "__main__":
    try:
        # Prepare Datasets
        logger.info(f"Preparing datasets...")
        source_data = prepare_datasets()
        train_dataset, valid_dataset = train_valid_split(source_data)
        logger.info(
            f"Prepared datasets with {len(train_dataset)} training samples and {len(valid_dataset)} validation samples."
        )

        # Compute Class Weights
        class_weights = compute_class_weights(train_dataset, N_CLASSES)
        logger.info(f"Computed class weights: {class_weights.cpu().numpy()}")

        # Setup Hyperparameter Optimization
        experiment = setup_hyperparameter_optimization()

        # Run Hyperparameter Search
        run_hyperparameter_search(experiment)

        logger.info("Hyperparameter optimization completed successfully.")

    except Exception as main_e:
        logger.exception(f"An error occurred during the execution: {main_e}")
