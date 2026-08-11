"""BioReactTokenizer: combined reaction SMILES + amino acid pair tokenizer.

Pair format: <cls> reaction_tokens | AA-prefixed_sequence_tokens <eos>

The source Molformer vocabulary is never modified. BioReactTokenizer builds an
in-memory combined vocabulary where enzyme residues use distinct IDs such as
AA_A and AA_C, so shared symbols like C/N/O have different IDs in reactions and
protein sequences.
"""
import json
import re
from pathlib import Path
from typing import Optional

MOLFORMER_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>|>|\*|\$|\%[0-9]{2}|[0-9])"
)

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWYUBZOX")
AA_PREFIX = "AA_"
SPECIAL_TOKENS = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>", "|"]

_DEFAULT_VOCAB = Path(__file__).resolve().parents[1] / "assets" / "molformer_vocab.json"


class BioReactTokenizer:
    """Tokenizer for (reaction SMILES, enzyme sequence) pairs."""

    def __init__(self, vocab_path: Optional[str] = None):
        path = Path(vocab_path) if vocab_path else _DEFAULT_VOCAB
        with open(path) as f:
            molformer_vocab: dict[str, int] = json.load(f)

        token_to_id: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            token_to_id[tok] = len(token_to_id)
        for tok in molformer_vocab:
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)
        for aa in AA_ALPHABET:
            tok = f"{AA_PREFIX}{aa}"
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)

        self.vocab_path = path
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id["<cls>"]

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id["<mask>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def pipe_token_id(self) -> int:
        return self.token_to_id["|"]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def save_vocab(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as handle:
            json.dump(self.token_to_id, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def tokenize_reaction(self, rxn: str) -> list[str]:
        tokens = MOLFORMER_REGEX.findall(rxn)
        if "".join(tokens) != rxn:
            raise ValueError(f"Reaction tokenization mismatch: {rxn!r}")
        return tokens

    def tokenize_sequence(self, seq: str) -> list[str]:
        return [f"{AA_PREFIX}{aa}" for aa in seq]

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        unk = self.unk_token_id
        return [self.token_to_id.get(t, unk) for t in tokens]

    def encode_pair(
        self,
        reaction: str,
        sequence: str,
        max_length: Optional[int] = None,
    ) -> dict:
        """Encode (reaction, sequence) -> <cls> A... | B... <eos>."""
        rxn_tokens = self.tokenize_reaction(reaction)
        aa_tokens = self.tokenize_sequence(sequence)

        ids = (
            [self.cls_token_id]
            + self.tokens_to_ids(rxn_tokens)
            + [self.pipe_token_id]
            + self.tokens_to_ids(aa_tokens)
            + [self.eos_token_id]
        )
        type_ids = [0] * (1 + len(rxn_tokens) + 1) + [1] * (len(aa_tokens) + 1)

        n_rxn = len(rxn_tokens)
        if max_length and len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.eos_token_id]
            type_ids = type_ids[: max_length - 1] + [type_ids[-1]]
            n_rxn = min(n_rxn, max(0, max_length - 3))

        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "token_type_ids": type_ids,
            "n_rxn_tokens": n_rxn,
        }

    def batch_encode_pairs(
        self,
        reactions: list[str],
        sequences: list[str],
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> dict:
        encoded = [
            self.encode_pair(r, s, max_length=max_length)
            for r, s in zip(reactions, sequences)
        ]
        if not padding:
            return {k: [e[k] for e in encoded] for k in encoded[0]}

        max_len = max(len(e["input_ids"]) for e in encoded)
        if max_len % 8 != 0:
            max_len = ((max_len + 7) // 8) * 8

        input_ids, attn_masks, type_ids_batch = [], [], []
        for e in encoded:
            pad = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [self.pad_token_id] * pad)
            attn_masks.append(e["attention_mask"] + [0] * pad)
            type_ids_batch.append(e["token_type_ids"] + [0] * pad)

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "token_type_ids": type_ids_batch,
            "n_rxn_tokens": [e["n_rxn_tokens"] for e in encoded],
        }
