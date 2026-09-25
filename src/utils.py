import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from src.dataset import get_dataloaders
from src.train import (
    CONFIG_PROFILES,
    ConfigurableSeq2SeqTransformer,
    configure_cuda_for_rtx,
    get_amp_settings,
    get_device,
    greedy_decode,
)

# no. of insertions, deletions, or substitutions required to transform one string into another
def levenshtein_distance(a: str, b: str) -> int:
    """Compute edit distance with a small dynamic-programming fallback."""
    try:
        import Levenshtein
        return int(Levenshtein.distance(a, b))
    except ImportError:
        if len(a) < len(b):
            a, b = b, a
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current = [i]
            for j, cb in enumerate(b, 1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[j] + 1,
                        previous[j - 1] + int(ca != cb),
                    )
                )
            previous = current
        return previous[-1]


def bit_accuracy_8bit_ascii(target: str, prediction: str) -> float:
    """8-bit ASCII bit matching. Missing or excess bytes count as eight wrong bits."""
    target_bytes = target.encode("ascii")
    pred_bytes = prediction.encode("ascii", errors="replace")
    # Extra or missing characters still count against the full 8-bit denominator.
    denominator_bytes = max(len(target_bytes), len(pred_bytes))
    if denominator_bytes == 0:
        return 1.0
    hits = 0
    for target_byte, pred_byte in zip(target_bytes, pred_bytes):
        hits += 8 - (target_byte ^ pred_byte).bit_count()
    return hits / (8 * denominator_bytes)


def calculate_metrics(
    targets: List[str],
    predictions: List[str],
    tokenized_model: bool,
    evaluation_unit: str,
) -> Dict[str, float]:
    """Calculate chunk-level reconstruction metrics from decoded strings."""
    if not targets:
        return {}

    metrics: Dict[str, float] = {
        "evaluation_unit": evaluation_unit,
        "num_sequences": len(targets),
        "sequence_accuracy": sum(t == p for t, p in zip(targets, predictions)) / len(targets),
        "bit_level_accuracy": sum(
            bit_accuracy_8bit_ascii(t, p) for t, p in zip(targets, predictions)
        ) / len(targets),
        "levenshtein_distance": sum(
            levenshtein_distance(t, p) for t, p in zip(targets, predictions)
        ) / len(targets),
    }

    if tokenized_model:
        import sacrebleu
        from rouge_score import rouge_scorer

        sentence_bleu_scores = [
            sacrebleu.sentence_bleu(p, [t]).score
            for t, p in zip(targets, predictions)
        ]
        metrics["sentence_bleu"] = sum(sentence_bleu_scores) / len(sentence_bleu_scores)
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        metrics["rouge_l"] = sum(
            scorer.score(t, p)["rougeL"].fmeasure
            for t, p in zip(targets, predictions)
        ) / len(targets)
    return metrics


def reconstruct_full_lines(records: List[Dict]) -> Tuple[List[str], List[str]]:
    """Join non-overlapping chunk predictions back into their original lines."""
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["line_index"])].append(record)

    full_targets: List[str] = []
    full_predictions: List[str] = []
    # Chunks are concatenated in their original byte order for the secondary full-line view.
    for line_index in sorted(grouped):
        pieces = sorted(grouped[line_index], key=lambda x: int(x["chunk_index"]))
        full_targets.append("".join(piece["target"] for piece in pieces))
        full_predictions.append("".join(piece["prediction"] for piece in pieces))
    return full_targets, full_predictions


def print_metrics(title: str, metrics: Dict[str, float], tokenized_model: bool):
    """Print a compact metric summary for one evaluation view."""
    print(title)
    print(f"  Sequences: {metrics['num_sequences']}")
    print(f"  Sequence Accuracy: {100*metrics['sequence_accuracy']:.2f}%")
    print(f"  Bit-Level Accuracy: {100*metrics['bit_level_accuracy']:.2f}%")
    print(f"  Mean Levenshtein Distance: {metrics['levenshtein_distance']:.2f}")
    if tokenized_model:
        print(f"  Mean Sentence BLEU: {metrics['sentence_bleu']:.2f}")
        print(f"  Mean ROUGE-L: {100*metrics['rouge_l']:.2f}%")
    else:
        print("  Sentence BLEU: N/A (C5 token-free)")
        print("  ROUGE-L: N/A (C5 token-free)")


def model_from_checkpoint(checkpoint, config_name: str, device: torch.device):
    """Recreate a model from checkpoint metadata and load its saved weights."""
    cfg = CONFIG_PROFILES[config_name]
    mc = checkpoint["model_config"]
    model = ConfigurableSeq2SeqTransformer(
        src_vocab_size=int(mc["src_vocab_size"]),
        tgt_vocab_size=int(mc["tgt_vocab_size"]),
        d_model=int(mc["d_model"]),
        num_heads=int(mc["n_heads"]),
        num_kv_heads=cfg["num_kv_heads"],
        num_encoder_layers=int(mc["n_layers"]),
        num_decoder_layers=int(mc["n_layers"]),
        d_ff=int(mc["d_ff"]),
        dropout=float(mc["dropout"]),
        max_len=int(mc["max_seq_len"]),
        pos_encoding=cfg["pos_encoding"],
        attn_type=cfg["attn_type"],
        norm_type=cfg["norm_type"],
        is_token_free=cfg["is_token_free"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_one(config_name: str, args, device: torch.device):
    """Greedy-decode one configuration and save chunk-level and reconstructed metrics."""
    checkpoint_path = os.path.join("outputs", f"best_{config_name.lower()}_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {config_name}: missing {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CONFIG_PROFILES[config_name]
    dc = checkpoint["data_config"]
    tc = checkpoint["training_config"]

    _, _, test_loader, _, tgt_tok, meta = get_dataloaders(
        cipher_path=dc.get("cipher_path", args.cipher_path),
        plain_path=dc.get("plain_path", args.plain_path),
        is_token_free=cfg["is_token_free"],
        batch_size=64,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        chunk_bytes=int(dc.get("chunk_chars", 64)),
        stride_bytes=int(dc.get("stride_chars", 64)),
        max_seq_len=int(checkpoint["model_config"].get("max_seq_len", 1024)),
        src_vocab_size=int(dc.get("src_vocab_size", 2048)),
        tgt_vocab_size=int(dc.get("vocab_size", 4096)),
        tokenizer_train_chunks=int(dc.get("tokenizer_train_chunks", 4000)),
        seed=int(tc.get("seed", 42)),
        rebuild_tokenizers=False,
        cipher_tokenizer_path=dc.get("cipher_tokenizer_path", "outputs/cipher_tokenizer.json"),
        plain_tokenizer_path=dc.get("tokenizer_path", "outputs/tokenizer.json"),
    )

    model = model_from_checkpoint(checkpoint, config_name, device)
    amp_choice = tc.get("amp_dtype", "fp16") if args.amp_dtype == "checkpoint" else args.amp_dtype
    amp_enabled, amp_dtype, _ = get_amp_settings(device, amp_choice)
    decode_cap = args.max_decode_len if args.max_decode_len > 0 else int(
        dc.get("generation_max_len", meta["generation_max_len"])
    )

    chunk_chars = int(dc.get("chunk_chars", 64))
    print("=" * 72)
    print(f"Evaluating {config_name}: {cfg['description']}")
    print(
        f"Evaluation setup: independent {chunk_chars}-character chunks "
        f"({chunk_chars*8} cipher bits for a full ASCII chunk) | "
        f"batch={args.eval_batch_size} | greedy cap={decode_cap}"
    )
    print("=" * 72)

    targets: List[str] = []
    predictions: List[str] = []
    records: List[Dict] = []
    eos_hits = 0
    generated_lengths = []

    # Decode each test chunk independently, matching the unit used during training.
    for batch_index, batch in enumerate(test_loader):
        src = batch["src"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        ids = greedy_decode(
            model,
            src,
            src_mask,
            decode_cap,
            tgt_tok.bos_token_id,
            tgt_tok.eos_token_id,
            tgt_tok.pad_token_id,
            device,
            amp_enabled,
            amp_dtype,
        )

        for row in range(src.size(0)):
            row_ids = ids[row].tolist()
            eos_position = None
            if tgt_tok.eos_token_id in row_ids[1:]:
                eos_hits += 1
                eos_position = row_ids[1:].index(tgt_tok.eos_token_id) + 1
            generated_lengths.append(
                eos_position if eos_position is not None else max(0, len(row_ids) - 1)
            )
            prediction = tgt_tok.decode(row_ids, skip_special_tokens=True)
            target = batch["target_texts"][row]
            targets.append(target)
            predictions.append(prediction)
            records.append(
                {
                    "index": len(records),
                    "line_index": int(batch["line_indices"][row]),
                    "chunk_index": int(batch["chunk_indices"][row]),
                    "byte_start": int(batch["byte_starts"][row]),
                    "target": target,
                    "prediction": prediction,
                    "exact": target == prediction,
                    "bit_accuracy": bit_accuracy_8bit_ascii(target, prediction),
                }
            )

        if args.progress_every > 0 and (batch_index + 1) % args.progress_every == 0:
            done = min((batch_index + 1) * args.eval_batch_size, len(test_loader.dataset))
            print(f"  decoded {done}/{len(test_loader.dataset)} chunks")

    tokenized_model = not cfg["is_token_free"]
    chunk_metrics = calculate_metrics(
        targets,
        predictions,
        tokenized_model=tokenized_model,
        evaluation_unit=f"independent_{chunk_chars}_character_chunks",
    )
    chunk_metrics["eos_rate"] = eos_hits / max(1, len(targets))
    chunk_metrics["avg_generated_length"] = sum(generated_lengths) / max(1, len(generated_lengths))

    # Keep the reconstructed-line scores separate from the main chunk-level metrics.
    full_targets, full_predictions = reconstruct_full_lines(records)
    full_line_metrics = calculate_metrics(
        full_targets,
        full_predictions,
        tokenized_model=tokenized_model,
        evaluation_unit="reconstructed_original_test_lines_from_nonoverlapping_chunks",
    )

    print_metrics("Chunk-level test metrics", chunk_metrics, tokenized_model)
    print(f"  EOS Rate: {100*chunk_metrics['eos_rate']:.2f}%")
    print(f"  Average Generated Tokens/Bytes: {chunk_metrics['avg_generated_length']:.1f}")
    print()
    print_metrics("SECONDARY / reconstructed full-line metrics", full_line_metrics, tokenized_model)

    metrics = {
        "primary_chunk_metrics": chunk_metrics,
        "secondary_full_line_metrics": full_line_metrics,
        "notes": {
            "evaluation_unit": f"independent_{chunk_chars}_character_chunks",
            "chunk_chars": chunk_chars,
            "stride_chars": int(dc.get("stride_chars", 64)),
            "greedy_decoding": True,
        },
    }

    os.makedirs("outputs", exist_ok=True)
    metrics_path = os.path.join("outputs", f"metrics_{config_name.lower()}.json")
    predictions_path = os.path.join("outputs", f"predictions_{config_name.lower()}.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(predictions_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Saved {metrics_path}")
    print(f"Saved {predictions_path}\n")
    return metrics


def parse_args():
    """Parse command-line options for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--cipher_path", default="data/brown_cipher.txt")
    parser.add_argument("--plain_path", default="data/brown_plain.txt")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_decode_len", type=int, default=0)
    parser.add_argument("--amp_dtype", choices=["checkpoint", "fp16", "bf16", "none"], default="checkpoint")
    parser.add_argument("--progress_every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_cuda_for_rtx()
    device = get_device()
    print(f"--> Evaluation device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else device}")
    for config_name in args.configs:
        if config_name not in CONFIG_PROFILES:
            print(f"Unknown config {config_name}; skipping")
            continue
        evaluate_one(config_name, args, device)
