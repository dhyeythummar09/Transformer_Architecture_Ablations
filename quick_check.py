from pathlib import Path

import torch

from src.dataset import (
    ByteTokenizer,
    ChunkedParallelDataset,
    ScratchBPETokenizer,
    Seq2SeqCollator,
    load_parallel_files,
    make_aligned_chunks,
)
from src.train import build_model, configure_cuda_for_rtx, get_device, parse_args


def tiny_batch(config_name: str):
    cipher_lines, plain_lines = load_parallel_files("data/brown_cipher.txt", "data/brown_plain.txt")
    chunks = make_aligned_chunks(cipher_lines[:2], plain_lines[:2], chunk_bytes=64, stride_bytes=64)[:2]
    is_token_free = config_name == "C5"

    if is_token_free:
        src_tok = ByteTokenizer()
        tgt_tok = ByteTokenizer()
    else:
        cipher_path = Path("outputs/cipher_tokenizer.json")
        plain_path = Path("outputs/tokenizer.json")
        if not cipher_path.exists() or not plain_path.exists():
            raise FileNotFoundError(
                "Cached BPE files are missing. Use the supplied outputs/cipher_tokenizer.json "
                "and outputs/tokenizer.json, or run C1 once with --rebuild_tokenizers."
            )
        src_tok = ScratchBPETokenizer.load(str(cipher_path))
        tgt_tok = ScratchBPETokenizer.load(str(plain_path))

    dataset = ChunkedParallelDataset(
        chunks,
        src_tok,
        tgt_tok,
        is_token_free=is_token_free,
        max_seq_len=1024,
    )
    collator = Seq2SeqCollator(
        src_pad_id=src_tok.pad_token_id,
        tgt_pad_id=tgt_tok.pad_token_id,
        tgt_bos_id=tgt_tok.bos_token_id,
        is_token_free=is_token_free,
        patch_size=4,
    )
    batch = collator([dataset[0], dataset[1]])
    return batch, src_tok, tgt_tok


def check(config_name: str, device: torch.device):
    args = parse_args()
    args.config_name = config_name
    batch, src_tok, tgt_tok = tiny_batch(config_name)
    model = build_model(config_name, src_tok.vocab_size, tgt_tok.vocab_size, args).to(device)
    with torch.inference_mode():
        logits = model(
            batch["src"].to(device),
            batch["tgt_input"].to(device),
            src_mask=batch["src_mask"].to(device),
            tgt_mask=batch["tgt_mask"].to(device),
        )
    print(
        f"{config_name}: src={tuple(batch['src'].shape)} "
        f"tgt={tuple(batch['tgt_input'].shape)} logits={tuple(logits.shape)} OK"
    )


if __name__ == "__main__":
    configure_cuda_for_rtx()
    device = get_device()
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA capability:", torch.cuda.get_device_capability(0))
    check("C1", device)
    check("C5", device)
    print("Quick check passed.")
