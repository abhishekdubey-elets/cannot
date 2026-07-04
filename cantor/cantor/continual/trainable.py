"""TrainableLM — a fine-tunable backend for the parametric arm (Phase 1).

Interface the RealLoRALearner drives:
    generate(prompt) -> str            rollout / probe
    train_lora(batch, steps, lr)       one SFT update on (prompt, completion) pairs
    snapshot() / restore(snap)         for eval-gate rollback

Two implementations:
  HFTrainableLM   real: transformers + peft LoRA over an open-weights model.
                  Import-guarded; this is the Phase-1 production backend.
  FakeTrainableLM offline test double: a deterministic "model" whose per-schema
                  competence rises after it is trained on that schema's examples,
                  and whose general capability drifts unless the batch contains
                  base-distribution anchors. Lets the FULL Phase-1 control flow
                  (batch build, train, eval-gate, rollback, re-eval) run and be
                  tested with no GPU. It is a test double, not a result.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

P_GEN = 0.85          # ceiling accuracy on the general probe when health = 1.0
P_DELIB = 0.90        # success of an EFFORTFUL (System-2, verified) online solve


def _draw(prompt: str) -> float:
    """Deterministic per-prompt pseudo-difficulty in [0,1)."""
    return (int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 10_000) / 10_000.0


def _tag(prompt: str) -> str:
    m = re.search(r"\[([^\]]+)\]", prompt)
    return m.group(1) if m else "?"


class TrainableLM:
    def generate(self, prompt: str, temperature: float = 0.0,
                 deliberate: bool = False) -> str: ...
    def train_lora(self, batch, steps: int = 20, lr: float = 1e-4) -> None: ...
    def snapshot(self): ...
    def restore(self, snap) -> None: ...


class FakeTrainableLM(TrainableLM):
    """Offline stand-in. `oracle` maps prompt -> correct answer (the latent
    capability); the fake only *applies* it on schemas it has been trained on."""

    def __init__(self, oracle: dict[str, str], base_success: dict[str, float],
                 master_at: int = 2, drift: float = 0.06, recovery: float = 0.9):
        """`drift` = general-capability erosion per schema-training update (H3);
        `recovery` = fraction of that drift undone by generative replay anchors
        (literature-grounded ~0.9). The residual anchored drift is DERIVED, not
        tuned, so no test-double parameter is co-fit to the go/no-go boundary."""
        self.oracle = oracle
        self.base_success = base_success
        self.master_at = master_at
        self.drift = drift
        self.recovery = recovery                     # kept so A9 can override drift
        self.anchor_drift = drift * (1.0 - recovery)
        self._exposure: Counter = Counter()
        self._mastered: set[str] = set()
        self._general = 1.0

    def _wrong(self, prompt: str) -> str:
        return str(int(self.oracle.get(prompt, "0")) + 7)

    def generate(self, prompt: str, temperature: float = 0.0,
                 deliberate: bool = False) -> str:
        """`deliberate=True` models an effortful, verified System-2 solve (high
        success) used for ONLINE solving; the fast eval path (deliberate=False)
        falls back to the schema's base competence unless the skill is mastered."""
        tag = _tag(prompt)
        if tag == "G":
            ok = _draw(prompt) < P_GEN * self._general
        elif tag in self._mastered:
            ok = True
        elif deliberate:
            ok = _draw(prompt) < P_DELIB
        else:
            ok = _draw(prompt) < self.base_success.get(tag, 0.3)
        return self.oracle.get(prompt, "0") if ok else self._wrong(prompt)

    def train_lora(self, batch, steps: int = 20, lr: float = 1e-4) -> None:
        has_anchor = any(_tag(p) == "G" for p, _ in batch)
        schemas = Counter(_tag(p) for p, _ in batch if _tag(p) != "G")
        for s, c in schemas.items():
            self._exposure[s] += c
            if self._exposure[s] >= self.master_at:
                self._mastered.add(s)
        if schemas:                       # a schema-training update perturbs general
            self._general = max(0.0, self._general -
                                (self.anchor_drift if has_anchor else self.drift))

    def snapshot(self):
        return (set(self._mastered), Counter(self._exposure), self._general)

    def restore(self, snap) -> None:
        m, e, g = snap
        self._mastered = set(m); self._exposure = Counter(e); self._general = g


class ToolFakeLM(FakeTrainableLM):
    """Phase-2 test double with PRIMITIVE-LEVEL mastery (streams.make_tool_stream).

    A task `[Fk|search+filter]` is solvable on the fast path when its primitives
    are mastered; PARTIAL mastery (some primitives shared with already-trained
    families) raises pre-training accuracy -> forward transfer (FWT > 0). This is
    the mechanism a shared-substructure (realistic) domain adds over S-CTRL.
    """

    def __init__(self, oracle, base_success, master_at: int = 2,
                 drift: float = 0.06, recovery: float = 0.9, ceiling: float = 0.85):
        super().__init__(oracle, base_success, master_at, drift, recovery)
        self.ceiling = ceiling
        self._prims: Counter = Counter()
        self._mastered_prims: set[str] = set()

    @staticmethod
    def _parse(prompt: str):
        raw = _tag(prompt)
        if "|" not in raw:
            return raw, set()
        schema, prims = raw.split("|", 1)
        return schema, set(prims.split("+"))

    def generate(self, prompt: str, temperature: float = 0.0,
                 deliberate: bool = False) -> str:
        schema, prims = self._parse(prompt)
        if schema == "G":
            ok = _draw(prompt) < P_GEN * self._general
        elif prims and prims <= self._mastered_prims:
            ok = True                                   # full mastery
        elif deliberate:
            ok = _draw(prompt) < P_DELIB
        else:
            frac = len(prims & self._mastered_prims) / len(prims) if prims else 0.0
            base = self.base_success.get(schema, 0.25)
            ok = _draw(prompt) < base + frac * (self.ceiling - base)   # partial -> FWT
        return self.oracle.get(prompt, "0") if ok else self._wrong(prompt)

    def train_lora(self, batch, steps: int = 20, lr: float = 1e-4) -> None:
        has_anchor = any(_tag(p) == "G" for p, _ in batch)
        seen: Counter = Counter()
        for p, _ in batch:
            schema, prims = self._parse(p)
            if schema == "G":
                continue
            for pr in prims:
                seen[pr] += 1
        for pr, c in seen.items():
            self._prims[pr] += c
            if self._prims[pr] >= self.master_at:
                self._mastered_prims.add(pr)
        if seen:
            self._general = max(0.0, self._general -
                                (self.anchor_drift if has_anchor else self.drift))

    def snapshot(self):
        return (set(self._mastered_prims), Counter(self._prims), self._general)

    def restore(self, snap) -> None:
        m, e, g = snap
        self._mastered_prims = set(m); self._prims = Counter(e); self._general = g


class HFTrainableLM(TrainableLM):  # pragma: no cover - needs GPU + torch/peft
    """Real QLoRA backend for the `--real` path. Runs on a Lightning.ai Studio,
    Kaggle 2xT4, Colab, or any CUDA box — NOT in the offline harness env.

    Requires: torch, transformers, peft, bitsandbytes, accelerate.
    Defaults are sized for a single 16 GB T4 via 4-bit NF4 QLoRA.

    UNTESTED in the authoring environment (no GPU/deps here). Run the offline
    smoke first (`python run_phase1.py`), then this via `--real`. Verify a tiny
    end-to-end pass before the full sweep (see README "Running --real").

    Implements the eval-gate rollback contract by snapshotting ONLY the LoRA
    adapter params (a few MB), so promote-or-rollback is cheap.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                 lora_rank: int = 16, lora_alpha: int = 32,
                 load_in_4bit: bool = True, max_new_tokens: int = 8):
        import os
        try:
            import torch
            from peft import (LoraConfig, get_peft_model,
                              prepare_model_for_kbit_training)
            from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                      BitsAndBytesConfig)
        except ImportError as e:
            raise ImportError("pip install torch transformers peft bitsandbytes "
                              "accelerate for HFTrainableLM") from e

        n_gpu = torch.cuda.device_count()
        if n_gpu == 0:
            raise RuntimeError(
                "HFTrainableLM requires a CUDA GPU, but torch sees none. If your "
                "traceback path is C:\\Users\\... you are running on your local "
                "machine, NOT the GPU server — SSH into the VM and run there.")

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        # T4 (Turing) has no bf16; auto-pick the supported compute dtype.
        self._dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Env override lets big GPUs (H100/A100) skip quantization: CANTOR_4BIT=0
        load_in_4bit = load_in_4bit and os.environ.get("CANTOR_4BIT", "1") != "0"

        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        quant = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=self._dtype,
                                    bnb_4bit_use_double_quant=True)
                 if load_in_4bit else None)
        # Force the whole model onto one GPU. device_map='auto' silently offloads
        # to CPU/disk when VRAM is tight, and bitsandbytes 4-bit then aborts with
        # "Some modules are dispatched on the CPU or the disk". Single-GPU pin
        # avoids that; multi-GPU (e.g. a 32B shard) falls back to 'auto'.
        device_map = {"": 0} if n_gpu == 1 else "auto"
        load_kw = dict(quantization_config=quant, device_map=device_map)
        try:                                        # transformers renamed torch_dtype -> dtype
            base = AutoModelForCausalLM.from_pretrained(model_name, dtype=self._dtype, **load_kw)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self._dtype, **load_kw)
        if load_in_4bit:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        else:                                       # full-precision path (no bitsandbytes)
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, cfg)

    # -- prompting ----------------------------------------------------------- #
    def _render(self, prompt: str, deliberate: bool) -> str:
        sys = ("Solve the task. Reason briefly, then give the final integer."
               if deliberate else "Answer with only the final integer.")
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, prompt: str, temperature: float = 0.0,
                 deliberate: bool = False) -> str:
        torch = self._torch
        self.model.eval()
        self.model.config.use_cache = True
        text = self._render(prompt, deliberate)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        budget = (self.max_new_tokens * 24) if deliberate else self.max_new_tokens
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=budget, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True)

    # -- one night's SFT update on (prompt, completion) pairs ----------------- #
    def train_lora(self, batch, steps: int = 20, lr: float = 1e-4) -> None:
        torch = self._torch
        if not batch:
            return
        self.model.train()
        self.model.config.use_cache = False
        opt = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr)
        for step in range(steps):
            prompt, completion = batch[step % len(batch)]
            rendered = self._render(prompt, deliberate=False)
            p_ids = self.tok(rendered, return_tensors="pt").input_ids
            full = self.tok(rendered + str(completion) + self.tok.eos_token,
                            return_tensors="pt").input_ids.to(self.model.device)
            labels = full.clone()
            labels[0, :p_ids.shape[1]] = -100          # mask the prompt tokens
            loss = self.model(input_ids=full, labels=labels).loss
            loss.backward()
            opt.step(); opt.zero_grad()

    # -- eval-gate rollback: snapshot ONLY the LoRA adapter ------------------- #
    def snapshot(self):
        return {k: v.detach().cpu().clone()
                for k, v in self.model.named_parameters() if v.requires_grad}

    def restore(self, snap) -> None:
        torch = self._torch
        with torch.no_grad():
            for k, v in self.model.named_parameters():
                if k in snap:
                    v.copy_(snap[k].to(v.device))
