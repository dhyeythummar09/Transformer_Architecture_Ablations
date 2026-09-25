import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
SPECIAL_COUNT = 4
BYTE_VOCAB_SIZE = 260


def cipher_bits_to_bytes(bits: str) -> bytes:
    """Convert a string of binary digits into the corresponding raw bytes."""
    bits = bits.strip()
    if len(bits) % 8 != 0:
        raise ValueError(f"Cipher length {len(bits)} is not divisible by 8")
    if set(bits) - {"0", "1"}:
        raise ValueError("Cipher contains symbols other than 0 and 1")
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


class ByteTokenizer:
    """Raw-byte tokenizer used only by C5/BLT."""

    pad_token_id = PAD_ID
    bos_token_id = BOS_ID
    eos_token_id = EOS_ID
    unk_token_id = UNK_ID
    vocab_size = BYTE_VOCAB_SIZE

    def encode_bytes(self, raw: bytes, add_special_tokens: bool = True) -> List[int]:
        """Map raw bytes to token ids, optionally adding BOS and EOS."""
        ids = [SPECIAL_COUNT + int(b) for b in raw]
        if add_special_tokens:
            ids = [BOS_ID, *ids, EOS_ID]
        return ids

    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode normal text as UTF-8 bytes."""
        return self.encode_bytes(text.encode("ascii"), add_special_tokens)

    def encode_cipher(self, bits: str, add_special_tokens: bool = False) -> List[int]:
        """Pack binary ciphertext into bytes before encoding it."""
        return self.encode_bytes(cipher_bits_to_bytes(bits), add_special_tokens)

    def decode_bytes(self, ids: Sequence[int], skip_special_tokens: bool = True) -> bytes:
        """Recover raw bytes from token ids while ignoring special tokens."""
        out = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id == EOS_ID:
                break
            if token_id < SPECIAL_COUNT:
                if skip_special_tokens:
                    continue
                continue
            if SPECIAL_COUNT <= token_id < BYTE_VOCAB_SIZE:
                out.append(token_id - SPECIAL_COUNT)
        return bytes(out)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to UTF-8 text."""
        raw = self.decode_bytes(ids, skip_special_tokens)
        return raw.decode("ascii", errors="replace")


class ScratchBPETokenizer:
    """
    Fully from-scratch byte-pair encoder.

    For ciphertext the byte stream is the literal ASCII binary string, e.g.
    b"010011...". Therefore BPE learns variable-length bit substrings directly.
    For plaintext the byte stream is ordinary ASCII text.
    """

    pad_token_id = PAD_ID
    bos_token_id = BOS_ID
    eos_token_id = EOS_ID
    unk_token_id = UNK_ID

    def __init__(self, target_vocab_size: int, min_frequency: int = 2):
        """Create a scratch BPE tokenizer with a fixed target vocabulary size."""
        if target_vocab_size <= SPECIAL_COUNT:
            raise ValueError("target_vocab_size is too small")
        self.target_vocab_size = int(target_vocab_size)
        self.min_frequency = int(min_frequency)
        self.base_symbols: List[int] = []
        self.base_to_id: Dict[int, int] = {}
        self.token_bytes: Dict[int, bytes] = {}
        self.merges: List[Tuple[int, int, int]] = []
        self.merge_rank: Dict[Tuple[int, int], int] = {}
        self.merge_id: Dict[Tuple[int, int], int] = {}
        self.first_merge_id = SPECIAL_COUNT
        self.vocab_size = SPECIAL_COUNT
        self._cache: Dict[bytes, Tuple[int, ...]] = {}

    @staticmethod
    def _best_pair(pair_occurrences: Dict[Tuple[int, int], set]):
        """Return the most frequent adjacent symbol pair in the current corpus."""
        best_pair = None
        best_count = -1
        for pair, occurrences in pair_occurrences.items():
            count = len(occurrences)
            if count > best_count or (count == best_count and (best_pair is None or pair < best_pair)):
                best_pair = pair
                best_count = count
        return best_pair, best_count

    def train(self, corpus: Sequence[bytes], verbose: bool = True) -> None:
        """Learn BPE merges directly from the supplied byte sequences."""
        corpus = [bytes(x) for x in corpus if len(x) > 0]
        if not corpus:
            raise ValueError("Cannot train BPE on an empty corpus")

        # Start from the bytes that actually appear in the training sample.
        self.base_symbols = sorted({b for raw in corpus for b in raw})
        self.base_to_id = {
            b: SPECIAL_COUNT + i for i, b in enumerate(self.base_symbols)
        }
        self.token_bytes = {
            token_id: bytes([b]) for b, token_id in self.base_to_id.items()
        }
        self.first_merge_id = SPECIAL_COUNT + len(self.base_symbols)
        next_id = self.first_merge_id

        values: List[int] = []
        prev: List[int] = []
        nxt: List[int] = []
        alive: List[bool] = []
        pair_occurrences: Dict[Tuple[int, int], set] = defaultdict(set)

        position = 0
        for raw in corpus:
            start = position
            ids = [self.base_to_id[b] for b in raw]
            for j, token_id in enumerate(ids):
                values.append(token_id)
                prev.append(position - 1 if j > 0 else -1)
                nxt.append(position + 1 if j + 1 < len(ids) else -1)
                alive.append(True)
                position += 1
            for i in range(start, position - 1):
                pair_occurrences[(values[i], values[i + 1])].add(i)

        def pair_at(left_index: int):
            """Return the live symbol pair starting at an index, if one exists."""
            if left_index < 0 or left_index >= len(values) or not alive[left_index]:
                return None
            right_index = nxt[left_index]
            if right_index < 0 or not alive[right_index]:
                return None
            return values[left_index], values[right_index]

        def remove_at(left_index: int):
            """Remove one live pair occurrence from the frequency table."""
            pair = pair_at(left_index)
            if pair is None:
                return
            occurrences = pair_occurrences.get(pair)
            if occurrences is None:
                return
            occurrences.discard(left_index)
            if not occurrences:
                pair_occurrences.pop(pair, None)

        def add_at(left_index: int):
            """Add the current live pair at an index to the occurrence table."""
            pair = pair_at(left_index)
            if pair is not None:
                pair_occurrences[pair].add(left_index)

        # Each accepted pair creates one new symbol until the requested vocabulary is reached.
        merge_target = self.target_vocab_size - next_id
        for merge_number in range(max(0, merge_target)):
            if not pair_occurrences:
                break
            pair, frequency = self._best_pair(pair_occurrences)
            if pair is None or frequency < self.min_frequency:
                break

            left_token, right_token = pair
            new_id = next_id
            next_id += 1
            self.merges.append((left_token, right_token, new_id))
            self.token_bytes[new_id] = self.token_bytes[left_token] + self.token_bytes[right_token]

            # Merge every still-valid occurrence and update only its neighboring pairs.
            occurrences = sorted(list(pair_occurrences.get(pair, ())))
            for left_index in occurrences:
                if left_index < 0 or left_index >= len(values) or not alive[left_index]:
                    continue
                right_index = nxt[left_index]
                if (
                    right_index < 0
                    or not alive[right_index]
                    or values[left_index] != left_token
                    or values[right_index] != right_token
                ):
                    continue

                left_prev = prev[left_index]
                right_next = nxt[right_index]
                remove_at(left_prev)
                remove_at(left_index)
                remove_at(right_index)

                values[left_index] = new_id
                alive[right_index] = False
                nxt[left_index] = right_next
                if right_next >= 0:
                    prev[right_next] = left_index

                add_at(left_prev)
                add_at(left_index)

            if verbose and (
                merge_number == 0
                or (merge_number + 1) % 500 == 0
                or next_id == self.target_vocab_size
            ):
                print(
                    f"  BPE merge {merge_number + 1}/{merge_target} | "
                    f"pair frequency {frequency} | vocab {next_id}"
                )

        self.vocab_size = next_id
        self.merge_rank = {}
        self.merge_id = {}
        for rank, (left, right, new_id) in enumerate(self.merges):
            self.merge_rank[(left, right)] = rank
            self.merge_id[(left, right)] = new_id
        self._cache.clear()

        if self.vocab_size < self.target_vocab_size:
            print(
                f"WARNING: BPE stopped at vocab {self.vocab_size}; requested "
                f"{self.target_vocab_size}. Increase tokenizer training chunks if needed."
            )

    def _push_candidate(
        self,
        heap: List[Tuple[int, int, int]],
        values: List[int],
        nxt: List[int],
        alive: List[bool],
        left_index: int,
    ) -> None:
        """Add a merge candidate to the heap if it is still valid."""
        import heapq

        if left_index < 0 or not alive[left_index]:
            return
        right_index = nxt[left_index]
        if right_index < 0 or not alive[right_index]:
            return
        rank = self.merge_rank.get((values[left_index], values[right_index]))
        if rank is not None:
            heapq.heappush(heap, (rank, left_index, right_index))

    def _encode_uncached(self, raw: bytes) -> Tuple[int, ...]:
        """Apply learned merges to one byte sequence without using the cache."""
        import heapq

        if not raw:
            return tuple()
        values = [self.base_to_id.get(b, UNK_ID) for b in raw]
        n = len(values)
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1
        alive = [True] * n
        heap: List[Tuple[int, int, int]] = []

        for i in range(n - 1):
            self._push_candidate(heap, values, nxt, alive, i)

        while heap:
            rank, left_index, right_index = heapq.heappop(heap)
            if not alive[left_index] or not alive[right_index]:
                continue
            if nxt[left_index] != right_index:
                continue
            pair = values[left_index], values[right_index]
            if self.merge_rank.get(pair) != rank:
                continue

            values[left_index] = self.merge_id[pair]
            right_next = nxt[right_index]
            alive[right_index] = False
            nxt[left_index] = right_next
            if right_next >= 0:
                prev[right_next] = left_index

            self._push_candidate(heap, values, nxt, alive, prev[left_index])
            self._push_candidate(heap, values, nxt, alive, left_index)

        encoded: List[int] = []
        index = 0
        while index >= 0:
            if alive[index]:
                encoded.append(values[index])
            index = nxt[index]
        return tuple(encoded)

    def encode_bytes(self, raw: bytes, add_special_tokens: bool = True) -> List[int]:
        """Encode raw bytes with the learned BPE merges."""
        raw = bytes(raw)
        cached = self._cache.get(raw)
        if cached is None:
            cached = self._encode_uncached(raw)
            if len(self._cache) < 50000:
                self._cache[raw] = cached
        ids = list(cached)
        if add_special_tokens:
            ids = [BOS_ID, *ids, EOS_ID]
        return ids

    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode UTF-8 text with the learned BPE tokenizer."""
        return self.encode_bytes(text.encode("ascii"), add_special_tokens)

    def decode_bytes(self, ids: Sequence[int], skip_special_tokens: bool = True) -> bytes:
        """Reconstruct raw bytes from BPE token ids."""
        chunks: List[bytes] = []
        for token_id in ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id == EOS_ID:
                break
            if token_id < SPECIAL_COUNT:
                if skip_special_tokens:
                    continue
                continue
            piece = self.token_bytes.get(token_id)
            if piece is not None:
                chunks.append(piece)
        return b"".join(chunks)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode BPE token ids to UTF-8 text."""
        return self.decode_bytes(ids, skip_special_tokens).decode("ascii", errors="replace")

    def save(self, path: str) -> None:
        """Save the learned merge table and tokenizer settings as JSON."""
        payload = {
            "version": 3,
            "target_vocab_size": self.target_vocab_size,
            "min_frequency": self.min_frequency,
            "base_symbols": self.base_symbols,
            "merges": self.merges,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str):
        """Load a tokenizer previously saved with :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(
            target_vocab_size=int(payload["target_vocab_size"]),
            min_frequency=int(payload.get("min_frequency", 2)),
        )
        tok.base_symbols = [int(x) for x in payload["base_symbols"]]
        tok.base_to_id = {
            b: SPECIAL_COUNT + i for i, b in enumerate(tok.base_symbols)
        }
        tok.token_bytes = {
            token_id: bytes([b]) for b, token_id in tok.base_to_id.items()
        }
        tok.first_merge_id = SPECIAL_COUNT + len(tok.base_symbols)
        tok.merges = [tuple(map(int, x)) for x in payload["merges"]]
        tok.merge_rank = {}
        tok.merge_id = {}
        next_id = tok.first_merge_id
        for rank, (left, right, new_id) in enumerate(tok.merges):
            tok.merge_rank[(left, right)] = rank
            tok.merge_id[(left, right)] = new_id
            tok.token_bytes[new_id] = tok.token_bytes[left] + tok.token_bytes[right]
            next_id = max(next_id, new_id + 1)
        tok.vocab_size = next_id
        return tok

    def stats(self, corpus: Sequence[bytes]) -> Dict[str, float]:
        """Measure compression, merged-token usage, and unknown-token rate."""
        atoms = 0
        tokens = 0
        merged_tokens = 0
        unks = 0
        for raw in corpus:
            ids = self.encode_bytes(raw, add_special_tokens=False)
            atoms += len(raw)
            tokens += len(ids)
            merged_tokens += sum(i >= self.first_merge_id for i in ids)
            unks += sum(i == UNK_ID for i in ids)
        return {
            "atoms": float(atoms),
            "tokens": float(tokens),
            "compression": atoms / max(tokens, 1),
            "merged_fraction": merged_tokens / max(tokens, 1),
            "unk_fraction": unks / max(tokens, 1),
        }


class ChunkedParallelDataset(Dataset):
    """Store aligned ciphertext/plaintext chunks and their metadata."""

    @classmethod
    def from_samples(cls, samples, is_token_free: bool):
        """Build a dataset directly from pre-encoded samples."""
        obj = cls.__new__(cls)
        obj.samples = samples
        obj.is_token_free = bool(is_token_free)
        return obj

    def __init__(
        self,
        raw_chunks: Sequence[Dict[str, Any]],
        src_tokenizer,
        tgt_tokenizer,
        is_token_free: bool,
        max_seq_len: int,
    ):
        """Encode aligned chunks and keep the information needed for evaluation."""
        self.samples: List[Dict[str, Any]] = []
        self.is_token_free = bool(is_token_free)

        for sample_index, item in enumerate(raw_chunks):
            cipher_chunk = item["cipher_bits"]
            plain_bytes = item["plain_bytes"]

            if self.is_token_free:
                # C5 packs every 8 cipher bits into one raw byte. Source BOS/EOS are omitted
                # so a 64-byte plaintext chunk stays aligned with 64 source bytes (16 patches).
                src_ids = src_tokenizer.encode_cipher(cipher_chunk, add_special_tokens=False)
                tgt_ids = tgt_tokenizer.encode_bytes(plain_bytes, add_special_tokens=True)
            else:
                # C1-C4: learn BPE tokens directly from the binary string.
                src_ids = src_tokenizer.encode_bytes(cipher_chunk.encode("ascii"), add_special_tokens=True)
                tgt_ids = tgt_tokenizer.encode_bytes(plain_bytes, add_special_tokens=True)

            if len(src_ids) > max_seq_len:
                raise ValueError(
                    f"Source tokenized length {len(src_ids)} exceeds max_seq_len={max_seq_len}. "
                    "Do not silently truncate; increase max_seq_len."
                )
            if len(tgt_ids) > max_seq_len:
                raise ValueError(
                    f"Target tokenized length {len(tgt_ids)} exceeds max_seq_len={max_seq_len}. "
                    "Do not silently truncate; increase max_seq_len."
                )

            self.samples.append(
                {
                    "src_ids": torch.tensor(src_ids, dtype=torch.long),
                    "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
                    "cipher_bits": cipher_chunk,
                    "target_text": plain_bytes.decode("ascii"),
                    "target_bytes": plain_bytes,
                    "line_index": int(item["line_index"]),
                    "chunk_index": int(item["chunk_index"]),
                    "byte_start": int(item["byte_start"]),
                    "sample_index": sample_index,
                }
            )

    def __len__(self):
        """Return the number of aligned chunk pairs."""
        return len(self.samples)

    def __getitem__(self, index: int):
        """Return one encoded source-target chunk and its metadata."""
        return self.samples[index]


class Seq2SeqCollator:
    """Pad a batch and prepare masks and teacher-forcing targets."""
    def __init__(
        self,
        src_pad_id: int,
        tgt_pad_id: int,
        tgt_bos_id: int,
        is_token_free: bool,
        patch_size: int = 4,
    ):
        """Store token ids and patch settings used while collating batches."""
        self.src_pad_id = int(src_pad_id)
        self.tgt_pad_id = int(tgt_pad_id)
        self.tgt_bos_id = int(tgt_bos_id)
        self.is_token_free = bool(is_token_free)
        self.patch_size = int(patch_size)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad a list of examples and build the model-ready batch dictionary."""
        src = pad_sequence(
            [x["src_ids"] for x in batch],
            batch_first=True,
            padding_value=self.src_pad_id,
        )
        # Attention masks use True for real tokens and False for padding.
        src_mask = (src != self.src_pad_id).unsqueeze(1).unsqueeze(2)

        if not self.is_token_free:
            tgt = pad_sequence(
                [x["tgt_ids"] for x in batch],
                batch_first=True,
                padding_value=self.tgt_pad_id,
            )
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_pad_mask = (tgt_input != self.tgt_pad_id).unsqueeze(1).unsqueeze(2)
            length = tgt_input.size(1)
            causal = torch.tril(torch.ones(length, length, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
            tgt_mask = tgt_pad_mask & causal
        else:
            # Shift by one full patch so the decoder never sees bytes it is predicting.
            labels = [x["tgt_ids"][1:] for x in batch]  # remove BOS; retain EOS
            tgt_output = pad_sequence(labels, batch_first=True, padding_value=self.tgt_pad_id)
            extra = (-tgt_output.size(1)) % self.patch_size
            if extra:
                tgt_output = F.pad(tgt_output, (0, extra), value=self.tgt_pad_id)

            batch_size = tgt_output.size(0)
            first_patch = torch.full(
                (batch_size, self.patch_size),
                self.tgt_pad_id,
                dtype=torch.long,
            )
            first_patch[:, 0] = self.tgt_bos_id
            tgt_input = torch.cat(
                [first_patch, tgt_output[:, :-self.patch_size]],
                dim=1,
            )
            patch_len = math.ceil(tgt_input.size(1) / self.patch_size)
            tgt_mask = torch.tril(torch.ones(patch_len, patch_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)

        return {
            "src": src,
            "src_mask": src_mask,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
            "tgt_mask": tgt_mask,
            "target_texts": [x["target_text"] for x in batch],
            "target_bytes": [x["target_bytes"] for x in batch],
            "line_indices": [x["line_index"] for x in batch],
            "chunk_indices": [x["chunk_index"] for x in batch],
            "byte_starts": [x["byte_start"] for x in batch],
        }


def load_parallel_files(cipher_path: str, plain_path: str) -> Tuple[List[str], List[str]]:
    """Read ciphertext and plaintext files and verify that their line counts match."""
    cipher = Path(cipher_path).read_text(encoding="utf-8").splitlines()
    plain = Path(plain_path).read_text(encoding="utf-8").splitlines()
    if len(cipher) != len(plain):
        raise ValueError("Cipher/plain line counts differ")
    return cipher, plain


def split_parallel(cipher: Sequence[str], plain: Sequence[str]):
    """Split aligned examples into deterministic train, validation, and test partitions."""
    n = len(cipher)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return {
        "train": (list(cipher[:train_end]), list(plain[:train_end])),
        "val": (list(cipher[train_end:val_end]), list(plain[train_end:val_end])),
        "test": (list(cipher[val_end:]), list(plain[val_end:])),
    }


def make_aligned_chunks(
    cipher_lines: Sequence[str],
    plain_lines: Sequence[str],
    chunk_bytes: int = 64,
    stride_bytes: int = 64,
) -> List[Dict[str, Any]]:
    """Split each plaintext line and its ciphertext into matching fixed-size chunks."""
    if chunk_bytes <= 0 or stride_bytes <= 0:
        raise ValueError("chunk_bytes and stride_bytes must be positive")
    chunks: List[Dict[str, Any]] = []

    for line_index, (cipher, plain) in enumerate(zip(cipher_lines, plain_lines)):
        plain_bytes = plain.encode("ascii")
        if len(cipher) != 8 * len(plain_bytes):
            raise ValueError(
                f"Line {line_index}: expected cipher length {8*len(plain_bytes)}, got {len(cipher)}"
            )
        if set(cipher) - {"0", "1"}:
            raise ValueError(f"Line {line_index}: cipher is not binary")

        # Plaintext byte offsets map directly to 8x-long slices in the binary ciphertext.
        chunk_index = 0
        for start in range(0, len(plain_bytes), stride_bytes):
            end = min(start + chunk_bytes, len(plain_bytes))
            if end <= start:
                continue
            chunks.append(
                {
                    "cipher_bits": cipher[start * 8:end * 8],
                    "plain_bytes": plain_bytes[start:end],
                    "line_index": line_index,
                    "chunk_index": chunk_index,
                    "byte_start": start,
                }
            )
            chunk_index += 1
            if end == len(plain_bytes):
                break
    return chunks


def _train_or_load_bpe_pair(
    train_chunks: Sequence[Dict[str, Any]],
    cipher_tokenizer_path: str,
    plain_tokenizer_path: str,
    src_vocab_size: int,
    tgt_vocab_size: int,
    tokenizer_train_chunks: int,
    seed: int,
    rebuild: bool,
):
    """Load cached BPE tokenizers or train a matching ciphertext/plaintext pair."""
    if (
        os.path.exists(cipher_tokenizer_path)
        and os.path.exists(plain_tokenizer_path)
        and not rebuild
    ):
        print(f"Loading cipher BPE: {cipher_tokenizer_path}")
        src_tok = ScratchBPETokenizer.load(cipher_tokenizer_path)
        print(f"Loading plaintext BPE: {plain_tokenizer_path}")
        tgt_tok = ScratchBPETokenizer.load(plain_tokenizer_path)
        return src_tok, tgt_tok

    # Use a fixed sample of training chunks so tokenizer training is reproducible.
    rng = random.Random(seed)
    indices = list(range(len(train_chunks)))
    rng.shuffle(indices)
    if tokenizer_train_chunks > 0:
        indices = indices[: min(tokenizer_train_chunks, len(indices))]

    source_corpus = [train_chunks[i]["cipher_bits"].encode("ascii") for i in indices]
    target_corpus = [train_chunks[i]["plain_bytes"] for i in indices]

    print(
        f"Training ciphertext BPE DIRECTLY on binary strings | "
        f"sampled chunks={len(indices)} | target vocab={src_vocab_size}"
    )
    src_tok = ScratchBPETokenizer(src_vocab_size, min_frequency=2)
    src_tok.train(source_corpus)
    src_tok.save(cipher_tokenizer_path)

    print(
        f"Training plaintext BPE | sampled chunks={len(indices)} | "
        f"target vocab={tgt_vocab_size}"
    )
    tgt_tok = ScratchBPETokenizer(tgt_vocab_size, min_frequency=2)
    tgt_tok.train(target_corpus)
    tgt_tok.save(plain_tokenizer_path)

    return src_tok, tgt_tok


def _verify_bpe_pair(
    src_tok: ScratchBPETokenizer,
    tgt_tok: ScratchBPETokenizer,
    chunks: Sequence[Dict[str, Any]],
    checks: int = 100,
) -> None:
    """Check that both BPE tokenizers round-trip representative chunks exactly."""
    checks = min(checks, len(chunks))
    for item in chunks[:checks]:
        src = item["cipher_bits"].encode("ascii")
        tgt = item["plain_bytes"]
        if src_tok.decode_bytes(src_tok.encode_bytes(src)) != src:
            raise RuntimeError("Cipher BPE round-trip failure")
        if tgt_tok.decode_bytes(tgt_tok.encode_bytes(tgt)) != tgt:
            raise RuntimeError("Plaintext BPE round-trip failure")
    print(f"BPE round-trip checks passed on {checks} chunks.")


def _loader_kwargs(num_workers: int) -> Dict[str, Any]:
    """Return common DataLoader options for the current worker count."""
    kwargs: Dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def get_dataloaders(
    cipher_path: str = "data/brown_cipher.txt",
    plain_path: str = "data/brown_plain.txt",
    is_token_free: bool = False,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    num_workers: int = 0,
    chunk_bytes: int = 64,
    stride_bytes: int = 64,
    max_seq_len: int = 1024,
    src_vocab_size: int = 2048,
    tgt_vocab_size: int = 4096,
    tokenizer_train_chunks: int = 4000,
    seed: int = 42,
    rebuild_tokenizers: bool = False,
    cipher_tokenizer_path: str = "outputs/cipher_tokenizer.json",
    plain_tokenizer_path: str = "outputs/tokenizer.json",
):
    """Build tokenizers, datasets, loaders, and metadata for one configuration."""
    cipher_lines, plain_lines = load_parallel_files(cipher_path, plain_path)
    splits = split_parallel(cipher_lines, plain_lines)

    raw = {}
    for split_name, (c, p) in splits.items():
        raw[split_name] = make_aligned_chunks(
            c,
            p,
            chunk_bytes=chunk_bytes,
            stride_bytes=stride_bytes,
        )

    if is_token_free:
        src_tok = ByteTokenizer()
        tgt_tok = ByteTokenizer()
    else:
        src_tok, tgt_tok = _train_or_load_bpe_pair(
            raw["train"],
            cipher_tokenizer_path,
            plain_tokenizer_path,
            src_vocab_size,
            tgt_vocab_size,
            tokenizer_train_chunks,
            seed,
            rebuild_tokenizers,
        )
        if src_tok.vocab_size != src_vocab_size:
            raise RuntimeError(
                f"Cipher tokenizer vocab={src_tok.vocab_size}, expected {src_vocab_size}. "
                "Delete outputs/cipher_tokenizer.json and rebuild with more tokenizer_train_chunks."
            )
        if tgt_tok.vocab_size != tgt_vocab_size:
            raise RuntimeError(
                f"Plain tokenizer vocab={tgt_tok.vocab_size}, expected {tgt_vocab_size}. "
                "Delete outputs/tokenizer.json and rebuild with more tokenizer_train_chunks."
            )
        _verify_bpe_pair(src_tok, tgt_tok, raw["val"], checks=100)

        src_stats = src_tok.stats(
            [x["cipher_bits"].encode("ascii") for x in raw["val"][:500]]
        )
        tgt_stats = tgt_tok.stats([x["plain_bytes"] for x in raw["val"][:500]])
        print(
            "Tokenizer stats | "
            f"cipher compression={src_stats['compression']:.2f}x, "
            f"merged={100*src_stats['merged_fraction']:.1f}%, "
            f"UNK={100*src_stats['unk_fraction']:.3f}% | "
            f"plain compression={tgt_stats['compression']:.2f}x, "
            f"merged={100*tgt_stats['merged_fraction']:.1f}%, "
            f"UNK={100*tgt_stats['unk_fraction']:.3f}%"
        )

    # Caching the encoded BPE chunks avoids repeating tokenization on later runs.
    encoded_cache_path = os.path.join(
        "outputs",
        f"tokenized_bpe_chunks_{chunk_bytes}_{stride_bytes}_{src_vocab_size}_{tgt_vocab_size}.pt",
    )
    if (
        not is_token_free
        and os.path.exists(encoded_cache_path)
        and not rebuild_tokenizers
    ):
        print(f"Loading cached tokenized chunks: {encoded_cache_path}")
        payload = torch.load(encoded_cache_path, map_location="cpu", weights_only=False)
        datasets = {
            name: ChunkedParallelDataset.from_samples(payload[name], is_token_free=False)
            for name in ("train", "val", "test")
        }
    else:
        datasets = {
            name: ChunkedParallelDataset(
                chunks,
                src_tok,
                tgt_tok,
                is_token_free=is_token_free,
                max_seq_len=max_seq_len,
            )
            for name, chunks in raw.items()
        }
        if not is_token_free:
            Path(encoded_cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {name: datasets[name].samples for name in ("train", "val", "test")},
                encoded_cache_path,
            )
            print(f"Saved tokenized chunk cache: {encoded_cache_path}")

    collator = Seq2SeqCollator(
        src_pad_id=src_tok.pad_token_id,
        tgt_pad_id=tgt_tok.pad_token_id,
        tgt_bos_id=tgt_tok.bos_token_id,
        is_token_free=is_token_free,
        patch_size=4,
    )
    kwargs = _loader_kwargs(num_workers)
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        collate_fn=collator,
        **kwargs,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **kwargs,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        **kwargs,
    )

    metadata = {
        "train_chunks": len(datasets["train"]),
        "val_chunks": len(datasets["val"]),
        "test_chunks": len(datasets["test"]),
        "chunk_bytes": int(chunk_bytes),
        "stride_bytes": int(stride_bytes),
        "generation_max_len": 96 if is_token_free else 128,
        "src_vocab_size": int(src_tok.vocab_size),
        "tgt_vocab_size": int(tgt_tok.vocab_size),
        "cipher_tokenizer_path": cipher_tokenizer_path if not is_token_free else None,
        "plain_tokenizer_path": plain_tokenizer_path if not is_token_free else None,
    }
    return train_loader, val_loader, test_loader, src_tok, tgt_tok, metadata
