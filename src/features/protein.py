from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import esm


DEFAULT_MAX_LENGTH = 500
DEFAULT_REPR_LAYER = 6


class ESMProteinEncoder:
    """
    Generate fixed protein representations using ESM-2.

    Model:
        esm2_t6_8M_UR50D

    Representation:
        Layer 6

    Pooling:
        Mean pooling over residue embeddings
    """

    def __init__(
        self,
        max_length: int = DEFAULT_MAX_LENGTH,
        representation_layer: int = DEFAULT_REPR_LAYER,
        device: torch.device | None = None,
    ) -> None:

        self.max_length = max_length
        self.representation_layer = representation_layer

        self.device = (
            device
            if device is not None
            else self._get_device()
        )

        (
            self.model,
            self.alphabet,
        ) = esm.pretrained.esm2_t6_8M_UR50D()

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self.batch_converter = (
            self.alphabet.get_batch_converter()
        )

    @staticmethod
    def _get_device() -> torch.device:
        """
        Select the best available PyTorch device.
        """

        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        if torch.backends.mps.is_available():
            return torch.device(
                "mps"
            )

        return torch.device(
            "cpu"
        )

    def prepare_sequence(
        self,
        sequence: str,
    ) -> str:
        """
        Validate, normalise and truncate a protein sequence.
        """

        if not isinstance(sequence, str):
            raise TypeError(
                "Protein sequence must be a string."
            )

        sequence = (
            sequence
            .strip()
            .upper()
        )

        # Remove whitespace only.
        sequence = "".join(
            sequence.split()
        )

        if not sequence:
            raise ValueError(
                "Protein sequence cannot be empty."
            )

        return sequence[
            :self.max_length
        ]

    def encode_batch(
        self,
        sequences: Sequence[str],
    ) -> np.ndarray:
        """
        Generate mean-pooled ESM-2 embeddings for a batch
        of protein sequences.

        Returns
        -------
        np.ndarray
            Shape: (number_of_sequences, 320)
        """

        prepared_sequences = [
            self.prepare_sequence(
                sequence
            )
            for sequence in sequences
        ]

        batch_data = [
            (
                f"protein_{index}",
                sequence,
            )
            for index, sequence
            in enumerate(
                prepared_sequences
            )
        ]

        _, _, tokens = (
            self.batch_converter(
                batch_data
            )
        )

        tokens = tokens.to(
            self.device
        )

        with torch.inference_mode():

            results = self.model(
                tokens,
                repr_layers=[
                    self.representation_layer
                ],
                return_contacts=False,
            )

        representations = (
            results["representations"][
                self.representation_layer
            ]
        )

        embeddings = []

        for index, sequence in enumerate(
            prepared_sequences
        ):

            sequence_length = len(
                sequence
            )

            # Position 0 = BOS.
            #
            # We select only actual amino-acid tokens,
            # excluding BOS, EOS and padding.
            residue_embeddings = (
                representations[
                    index,
                    1:sequence_length + 1,
                    :,
                ]
            )

            pooled_embedding = (
                residue_embeddings.mean(
                    dim=0
                )
            )

            embeddings.append(
                pooled_embedding
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

        return np.stack(
            embeddings,
            axis=0,
        )

    def encode(
        self,
        sequence: str,
    ) -> np.ndarray:
        """
        Encode one protein sequence.
        """

        return self.encode_batch(
            [sequence]
        )[0]