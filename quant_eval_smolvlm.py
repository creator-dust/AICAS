#!/usr/bin/env python
"""
Quantization accuracy evaluation for local SmolVLM2 models.

Supported strategies:
- apot   : APoT-like weight quantization + int8 activation fake quantization.
- bfp    : Block Floating Point weight quantization + BFP activation fake quantization.
- hybrid : APoT on attention linear layers + BFP on MLP linear layers.

This script is designed for quick PTQ-style accuracy comparison on a custom VQA dataset.
It does not rewrite model files on disk.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVLMModelClass
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVLMModelClass

@dataclass
class EvalSample:
    sample_id: str
    image_path: Path
    question: str
    answer: str
    choices: Optional[List[str]]

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]", "", text)
    return text.strip()

def letter_to_index(letter: str) -> Optional[int]:
    if not letter:
        return None
    letter = letter.strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None
    return ord(letter) - ord("A")

def extract_choice_letter(text: str) -> Optional[str]:
    text_up = text.upper()
    patterns = [
        r"(?:OPTION|ANSWER|CHOICE)\s*[:=]?\s*([A-Z])\b",
        r"^\s*\(?\s*([A-Z])\s*\)?\s*$",
        r"^\s*([A-Z])[\.\):\s]",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text_up)
        if m:
            return m.group(1)
    return None

def _pick(d: Dict, keys: Sequence[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def parse_sample(raw: Dict, root_dir: Path, idx: int) -> EvalSample:
    sample_id = str(_pick(raw, ["id", "sample_id", "qid"], f"sample_{idx}"))

    image_rel = _pick(raw, ["image", "image_path", "img", "img_path"])
    if image_rel is None:
        raise ValueError(f"Sample {sample_id} missing image path.")
    image_path = Path(image_rel)
    if not image_path.is_absolute():
        image_path = (root_dir / image_path).resolve()

    question = _pick(raw, ["question", "query", "prompt"])
    if question is None:
        raise ValueError(f"Sample {sample_id} missing question.")

    answer = _pick(raw, ["answer", "gt", "label", "target"])
    if answer is None:
        raise ValueError(f"Sample {sample_id} missing answer.")

    choices = _pick(raw, ["choices", "options", "candidates"], None)
    if choices is not None:
        if isinstance(choices, dict):
            ordered: List[str] = []
            for key in sorted(choices.keys()):
                ordered.append(str(choices[key]))
            choices = ordered
        elif isinstance(choices, list):
            choices = [str(x) for x in choices]
        else:
            raise ValueError(f"Sample {sample_id} has invalid choices format.")

    return EvalSample(
        sample_id=sample_id,
        image_path=image_path,
        question=str(question),
        answer=str(answer),
        choices=choices,
    )

def load_dataset(path: Path, max_samples: Optional[int] = None) -> List[EvalSample]:
    root_dir = path.parent.resolve()
    samples: List[EvalSample] = []

    if path.suffix.lower() == ".jsonl":
        # Use utf-8-sig to tolerate BOM created by some Windows editors/tools.
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {i + 1}: {exc.msg}") from exc
            samples.append(parse_sample(raw, root_dir, i))
            if max_samples and len(samples) >= max_samples:
                break
    elif path.suffix.lower() == ".json":
        raw_obj = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw_obj, list):
            raise ValueError("JSON dataset must be a list of records.")
        for i, raw in enumerate(raw_obj):
            if not isinstance(raw, dict):
                raise ValueError(f"Record {i} in dataset is not a JSON object.")
            samples.append(parse_sample(raw, root_dir, i))
            if max_samples and len(samples) >= max_samples:
                break
    else:
        raise ValueError("Dataset file must be .jsonl or .json")

    return samples

def build_prompt(question: str, choices: Optional[List[str]]) -> str:
    if not choices:
        return (
            "Answer the question based on the image. "
            "Keep the response short and precise.\n"
            f"Question: {question}"
        )

    option_lines = []
    for i, text in enumerate(choices):
        option_lines.append(f"{chr(ord('A') + i)}. {text}")
    joined = "\n".join(option_lines)
    return (
        "Choose one option based on the image. "
        "Reply with only the option letter (e.g., A).\n"
        f"Question: {question}\n"
        f"Options:\n{joined}"
    )

def build_apot_levels(bits: int = 4, terms: int = 2, max_power: int = 6) -> torch.Tensor:
    base = [2.0 ** (-p) for p in range(max_power + 1)]
    candidates = {0.0}

    for a in base:
        candidates.add(a)
    if terms >= 2:
        for a in base:
            for b in base:
                candidates.add(min(a + b, 1.0))

    cands = sorted(candidates)
    target_levels = 2 ** (bits - 1)
    targets = torch.linspace(0.0, 1.0, steps=target_levels) ** 2

    cands_t = torch.tensor(cands, dtype=torch.float32)
    picked = []
    for t in targets:
        idx = torch.argmin(torch.abs(cands_t - t)).item()
        picked.append(float(cands_t[idx].item()))

    picked = sorted(set(picked + [0.0, 1.0]))
    return torch.tensor(picked, dtype=torch.float32)

def quantize_abs_to_levels(x_abs: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    levels = levels.to(device=x_abs.device, dtype=x_abs.dtype)
    idx = torch.bucketize(x_abs, levels)
    idx = torch.clamp(idx, min=1, max=levels.numel() - 1)
    lower = levels[idx - 1]
    upper = levels[idx]
    choose_upper = (x_abs - lower) > (upper - x_abs)
    out = torch.where(choose_upper, upper, lower)
    return out

def apot_quantize_tensor(x: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    max_abs = x.detach().abs().max()
    if max_abs.item() == 0:
        return torch.zeros_like(x)

    normalized = (x / max_abs).clamp(-1, 1)
    q_abs = quantize_abs_to_levels(normalized.abs(), levels)
    q = normalized.sign() * q_abs
    return q * max_abs

def bfp_quantize_tensor(x: torch.Tensor, block_size: int = 32, mantissa_bits: int = 7) -> torch.Tensor:
    if x.numel() == 0:
        return x

    flat = x.detach().reshape(-1)
    n = flat.numel()
    rem = n % block_size
    if rem != 0:
        pad = block_size - rem
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)], dim=0)

    blocks = flat.reshape(-1, block_size)
    max_abs = blocks.abs().amax(dim=1, keepdim=True)
    nonzero = max_abs > 0

    shared_exp = torch.zeros_like(max_abs)
    # ====== 核心数学修复：+1.0 防止最大值溢出被腰斩 ======
    shared_exp[nonzero] = torch.floor(torch.log2(max_abs[nonzero])) + 1.0
    # ====================================================

    qmax = float((1 << (mantissa_bits - 1)) - 1)
    scale = torch.pow(2.0, shared_exp - (mantissa_bits - 1))
    scale = torch.where(nonzero, scale, torch.ones_like(scale))

    q = torch.round(blocks / scale).clamp(-qmax - 1.0, qmax)
    deq = q * scale
    deq = torch.where(nonzero, deq, torch.zeros_like(deq))

    out = deq.reshape(-1)[:n]
    return out.reshape_as(x)

def int8_fake_quant(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    max_abs = x.detach().abs().amax()
    scale = torch.clamp(max_abs / 127.0, min=eps)
    q = torch.round(x / scale).clamp(-127, 127)
    return q * scale

def bfp_fake_quant_activation(x: torch.Tensor, block_size: int, mantissa_bits: int) -> torch.Tensor:
    return bfp_quantize_tensor(x, block_size=block_size, mantissa_bits=mantissa_bits)

class ActivationHookManager:
    def __init__(self) -> None:
        self.handles: List = []

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

def apply_activation_hooks(
    model: nn.Module,
    mode: str,
    bfp_block: int,
    bfp_act_bits: int,
) -> ActivationHookManager:
    mgr = ActivationHookManager()

    def pre_hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        if mode == "int8":
            xq = int8_fake_quant(x)
        elif mode == "bfp":
            xq = bfp_fake_quant_activation(x, block_size=bfp_block, mantissa_bits=bfp_act_bits)
        else:
            xq = x
        new_inputs = (xq,) + tuple(inputs[1:])
        return new_inputs

    for m in model.modules():
        if isinstance(m, nn.Linear):
            mgr.handles.append(m.register_forward_pre_hook(pre_hook))

    return mgr

def is_attention_linear(name: str) -> bool:
    tokens = ["self_attn", "attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj"]
    return any(t in name for t in tokens)

def is_mlp_linear(name: str) -> bool:
    tokens = ["mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj"]
    return any(t in name for t in tokens)

def apply_weight_quantization(model: nn.Module, strategy: str, args: argparse.Namespace) -> Dict[str, int]:
    apot_levels = build_apot_levels(bits=args.apot_bits, terms=args.apot_terms, max_power=args.apot_max_power)
    touched = {"apot": 0, "bfp": 0, "skip": 0}
    if strategy == "fp32":
        return touched

    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if mod.weight is None:
                continue
            if ("lm_head" in name) or ("embed_tokens" in name):
                touched["skip"] += 1
                continue

            w = mod.weight.data

            if strategy == "apot":
                q = apot_quantize_tensor(w, apot_levels.to(w.device, w.dtype))
                mod.weight.data.copy_(q)
                touched["apot"] += 1
            elif strategy == "bfp":
                q = bfp_quantize_tensor(w, block_size=args.bfp_block, mantissa_bits=args.bfp_weight_bits)
                mod.weight.data.copy_(q)
                touched["bfp"] += 1
            elif strategy == "hybrid":
                if is_attention_linear(name):
                    q = apot_quantize_tensor(w, apot_levels.to(w.device, w.dtype))
                    mod.weight.data.copy_(q)
                    touched["apot"] += 1
                elif is_mlp_linear(name):
                    q = bfp_quantize_tensor(w, block_size=args.bfp_block, mantissa_bits=args.bfp_weight_bits)
                    mod.weight.data.copy_(q)
                    touched["bfp"] += 1
                else:
                    touched["skip"] += 1
            elif strategy != "fp32":
                raise ValueError(f"Unknown strategy: {strategy}")

    return touched

def model_generate(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        processor_kwargs={},
    )

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_only = output_ids[:, prompt_len:]
    text = processor.batch_decode(gen_only, skip_special_tokens=True)[0]
    return text

def score_prediction(pred: str, sample: EvalSample, open_ended_metric: str = "exact") -> bool:
    answer = sample.answer

    if sample.choices:
        pred_letter = extract_choice_letter(pred)
        ans_letter = extract_choice_letter(answer)
        
        # ====== 新增的修复逻辑开始 ======
        if pred_letter and not ans_letter:
            # 模型输出了字母，但标准答案是具体文本
            pred_idx = letter_to_index(pred_letter)
            if pred_idx is not None and 0 <= pred_idx < len(sample.choices):
                # 查字典：把模型选的字母换算成对应的文字
                predicted_text = sample.choices[pred_idx]
                return normalize_text(predicted_text) == normalize_text(answer)
        # ====== 新增的修复逻辑结束 ======

        if pred_letter and ans_letter:
            return pred_letter == ans_letter

        if ans_letter:
            ans_idx = letter_to_index(ans_letter)
            if ans_idx is not None and 0 <= ans_idx < len(sample.choices):
                answer = sample.choices[ans_idx]

        pred_norm = normalize_text(pred)
        ans_norm = normalize_text(answer)
        if pred_norm == ans_norm:
            return True

        for i, c in enumerate(sample.choices):
            c_norm = normalize_text(c)
            if pred_norm == c_norm:
                true_idx = None
                if ans_letter is not None:
                    true_idx = letter_to_index(ans_letter)
                if true_idx is not None:
                    return i == true_idx
                return c_norm == ans_norm
        return False

    pred_norm = normalize_text(pred)
    ans_norm = normalize_text(answer)

    if open_ended_metric == "exact":
        return pred_norm == ans_norm
    if open_ended_metric == "contains":
        return ans_norm in pred_norm
    raise ValueError(f"Unknown open_ended_metric: {open_ended_metric}")

def evaluate_one_strategy(
    strategy: str,
    samples: List[EvalSample],
    args: argparse.Namespace,
) -> Dict:
    device = torch.device(args.device)
    if args.model_dtype == "auto":
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    elif args.model_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    if device.type == "cpu" and dtype == torch.float16:
        print("[warn] float16 on CPU is very slow; switching to float32 automatically.")
        dtype = torch.float32

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoVLMModelClass.from_pretrained(
        args.model_dir,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    q_stat = apply_weight_quantization(model, strategy, args)
    act_hooks = None
    if args.enable_activation_fake_quant:
        if strategy == "apot":
            act_mode = "int8"
        else:
            act_mode = "bfp"

        act_hooks = apply_activation_hooks(
            model,
            mode=act_mode,
            bfp_block=args.bfp_block,
            bfp_act_bits=args.bfp_act_bits,
        )

    correct = 0
    total = 0
    details = []
    t0 = time.time()

    for s in samples:
        if not s.image_path.exists():
            if not args.skip_details:
                details.append(
                    {
                        "id": s.sample_id,
                        "error": f"image not found: {s.image_path}",
                        "correct": False,
                    }
                )
            total += 1
            continue

        image = Image.open(s.image_path).convert("RGB")
        prompt = build_prompt(s.question, s.choices)
        pred = model_generate(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

        is_ok = score_prediction(pred, s, open_ended_metric=args.open_ended_metric)
        correct += int(is_ok)
        total += 1

        if not args.skip_details:
            details.append(
                {
                    "id": s.sample_id,
                    "image": str(s.image_path),
                    "question": s.question,
                    "answer": s.answer,
                    "prediction": pred,
                    "correct": is_ok,
                }
            )

        if args.verbose:
            print(f"[{strategy}] {s.sample_id}: {'OK' if is_ok else 'NG'}")
            print(f"   [标准答案]: {s.answer}")
            print(f"   [模型预测]: {pred}")
        elif args.log_every > 0 and (total % args.log_every == 0):
            running_acc = correct / total if total > 0 else 0.0
            print(
                f"[{strategy}] progress {total}/{len(samples)} "
                f"acc={running_acc:.4f} last={'OK' if is_ok else 'NG'} id={s.sample_id}"
            )

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0.0

    if act_hooks is not None:
        act_hooks.close()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    return {
        "strategy": strategy,
        "num_samples": total,
        "num_correct": correct,
        "accuracy": acc,
        "elapsed_sec": elapsed,
        "quantized_layers": q_stat,
        "details": details,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate APoT/BFP/Hybrid quantization accuracy on SmolVLM2.")

    script_dir = Path(__file__).resolve().parent
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    parser.add_argument("--model-dir", type=Path, default=script_dir / "SmolVLM2_Weights")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to JSON/JSONL evaluation set.")
    parser.add_argument("--output", type=Path, default=script_dir / "quant_eval_report.json")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--model-dtype", type=str, choices=["auto", "float16", "float32"], default="auto")

    parser.add_argument("--strategies", nargs="+", default=["apot", "bfp", "hybrid"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--open-ended-metric", choices=["exact", "contains"], default="exact")

    parser.add_argument("--apot-bits", type=int, default=4)
    parser.add_argument("--apot-terms", type=int, default=2)
    parser.add_argument("--apot-max-power", type=int, default=6)

    parser.add_argument("--bfp-block", type=int, default=32)
    # ====== 解耦：拆分权重和激活的位宽 ======
    parser.add_argument("--bfp-weight-bits", type=int, default=4, help="Mantissa bits for BFP weights (simulate W4)")
    parser.add_argument("--bfp-act-bits", type=int, default=8, help="Mantissa bits for BFP activations (simulate A8)")
    # ======================================
    parser.add_argument("--enable-activation-fake-quant", action="store_true")

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N samples when not using --verbose. Set 0 to disable.")
    parser.add_argument("--skip-details", action="store_true", help="Do not save per-sample details into the output JSON report.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    def resolve_input_path(path: Path) -> Path:
        if path.is_absolute():
            return path
        # Keep current behavior first (relative to CWD), then fallback to script dir.
        if path.exists():
            return path.resolve()
        candidate = (script_dir / path).resolve()
        return candidate

    args.model_dir = resolve_input_path(args.model_dir)
    args.dataset = resolve_input_path(args.dataset)
    if not args.output.is_absolute():
        args.output = (script_dir / args.output).resolve()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    samples = load_dataset(args.dataset, max_samples=args.max_samples)
    if not samples:
        raise ValueError("Dataset is empty.")

    all_results = []
    for strategy in args.strategies:
        strategy = strategy.lower()
        if strategy not in {"fp32", "apot", "bfp", "hybrid"}:
            raise ValueError(f"Unsupported strategy: {strategy}")
        print(f"\n=== Running strategy: {strategy} ===")
        result = evaluate_one_strategy(strategy, samples, args)
        all_results.append(result)
        print(
            f"{strategy}: accuracy={result['accuracy']:.4f} "
            f"({result['num_correct']}/{result['num_samples']}), "
            f"time={result['elapsed_sec']:.1f}s"
        )

    summary = {
        "model_dir": str(args.model_dir.resolve()),
        "dataset": str(args.dataset.resolve()),
        "num_samples": len(samples),
        "strategies": args.strategies,
        "results": all_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved report to: {args.output}")

if __name__ == "__main__":
    main()
