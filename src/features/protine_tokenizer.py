from __future__ import annotations
import numpy as np

class ProtineTokenizer:

    AMINO_ACIDS = ("ACDEFGHIKLMNPQRSTVWY")

    PAD_INDEX = 0
    UNKNOWN_INDEX = 21

    def __init__(self, max_length: int = 500):

        if max_length <=0:
            raise ValueError("max_length must be greater than zero")
        
        self.max_length = max_length

        self.amino_acid_to_index = {
            amino_acid: index + 1
            for index, amino_acid in enumerate(self.AMINO_ACIDS)
        }

        self.vocab_size = (len(self.AMINO_ACIDS)+2)

    def encode(self, sequence: str):

        if not isinstance(sequence, str):
            raise TypeError("Protine sequence must be string")
        
        sequence = (sequence.strip().upper())

        sequence = "".join(sequence.split())

        if not sequence:
            raise ValueError("Protine sequence cannot be empty.")
        
        sequence = sequence[:self.max_length]

        encoded = np.full(
            self.max_length,
            self.PAD_INDEX,
            dtype=np.int64
        )

        for position, amino_acid in enumerate(sequence):
            encoded[position] = (
                self.amino_acid_to_index.get(
                    amino_acid,
                    self.UNKNOWN_INDEX,
                )
            )

        return encoded
    
    def transform(self, sequences):
        if len(sequences) == 0:
            raise ValueError(
                "sequences cannot be empty."
            )
        encoded_sequence = [
            self.encode(sequence)
            for sequence in sequences
        ]

        return np.stack(
            encoded_sequence,
            axis=0,
        )
    
    def __repr__(self) -> str:

        return (
            "ProteinTokenizer("
            f"max_length={self.max_length}, "
            f"vocab_size={self.vocab_size}"
            ")"
        )
