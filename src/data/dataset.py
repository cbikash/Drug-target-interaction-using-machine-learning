from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

import torch



# CUSTOM PYTORCH DATASET FOR DRUG–TARGET INTERACTION DATA
class DrugTargetDataset(Dataset):
    """
    Custom PyTorch Dataset for drug–target affinity regression.

    Each observation consists of:
        1. A numerical ligand representation.
        2. A numerical protein representation.
        3. A continuous binding-affinity target (pKi).

    The Dataset class provides indexed access to individual
    drug–target pairs and converts the stored NumPy arrays into
    PyTorch tensors suitable for model training and evaluation.
    """

    def __init__(self, lig, prot, y):
        """
        Initialise the dataset.

        Parameters
        ----------
        lig : array-like
            Numerical ligand representations corresponding to
            individual drug molecules.

        prot : array-like
            Numerical protein representations corresponding to
            the associated protein targets.

        y : array-like
            Continuous target values representing experimentally
            derived drug–target binding affinity expressed as pKi.
        """

        # Store the ligand representations. Each row corresponds
        # to one drug molecule in a drug–target interaction pair.
        self.lig = lig

        # Store the corresponding protein representations.
        # The ith protein representation must correspond to the
        # same interaction record as the ith ligand representation.
        self.prot = prot

        # Store the continuous affinity values used as the
        # regression targets during supervised model training.
        self.y = y


    def __len__(self):
        """
        Return the total number of drug–target interaction records.

        The number of target values is used because each affinity
        measurement corresponds to exactly one ligand–protein pair.
        """
        return len(self.y)


    def __getitem__(self, idx):
        """
        Retrieve and convert a single drug–target interaction sample.

        Parameters
        ----------
        idx : int
            Index of the requested observation.

        Returns
        -------
        ligand : torch.Tensor
            Ligand representation converted to 32-bit floating point.

        protein : torch.Tensor
            Protein representation converted to an integer tensor.

        target : torch.Tensor
            Experimental pKi value represented as a 32-bit
            floating-point tensor.
        """

        return (
            # Molecular features are represented as floating-point
            # values before being provided to the ligand encoder.
            torch.tensor(
                self.lig[idx],
                dtype=torch.float32
            ),

            # Protein representations are converted to integer
            # tensors because this implementation assumes that the
            # protein input contains tokenised amino-acid indices
            # that are subsequently processed by an embedding layer.
            torch.tensor(
                self.prot[idx],
                dtype=torch.long
            ),

            # The pKi response variable is represented as a
            # floating-point value because affinity prediction is
            # formulated as a continuous regression problem.
            torch.tensor(
                self.y[idx],
                dtype=torch.float32
            ),
        )