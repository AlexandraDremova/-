"""Дообучение одной бинарной пары на CAMeLBERT.  [torch/transformers — ленивый импорт]

Модуль импортируется без torch (его тянут только функции обучения/инференса),
поэтому чистая логика и тесты голосования работают в любом окружении.

Контракт наружу:
    train_pair(target, opponent, train_df, **cfg) -> model_path (str)
    PairClassifier.train(...) -> PairClassifier (с .train_info — сводка обучения пары)
    PairClassifier.predict_proba_batch(texts) -> [P(label=1 == target)]

Текст на вход приходит УЖЕ очищенным (диакритика/кашида сняты на стороне данных).
Здесь только токенизация CAMeLBERT.

Устойчивость (по итогам прогона 2026-10-26, где 3 из 45 пар ушли в NaN в bf16):
  * обучение в bf16; при первом нечисловом loss/градиенте пара переобучается с нуля в fp32;
  * клиппинг нормы градиента (MAX_GRAD_NORM);
  * взвешенный loss для несбалансированных пар (BALANCE_PAIRS), напр. 60 vs 14 сегментов;
  * инференс всегда в fp32 и с проверкой, что вероятности — числа.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

from aa import config
from aa.log import fmt_duration
from aa.model.sampling import pair_frame, pair_seed

log = logging.getLogger(__name__)
_DEVICE_REPORTED = False


class NonFiniteTraining(RuntimeError):
    """Loss, градиент или веса стали NaN/inf."""


class NonFinitePrediction(RuntimeError):
    """Обученная модель выдала NaN/inf-вероятности — её голос учитывать нельзя."""


def _params_finite(torch, model) -> bool:
    return bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all().item())


def _lazy():
    """Импортируем тяжёлое только когда реально обучаем/предсказываем."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as hf_logging
    hf_logging.set_verbosity_error()  # заглушить отчёт загрузки голов на каждой из 45 пар
    return torch, DataLoader, TensorDataset, AutoModelForSequenceClassification, AutoTokenizer


_TOKENIZERS: dict = {}


def _get_tokenizer(checkpoint: str):
    """Токенизатор грузится один раз на процесс. В прогоне 26.10 повторная загрузка на каждой
    из 45 пар (с сетевой проверкой HuggingFace) съедала ~7 минут, в худшем случае 126 с на пару."""
    if checkpoint not in _TOKENIZERS:
        _, _, _, _, AutoTokenizer = _lazy()
        _TOKENIZERS[checkpoint] = AutoTokenizer.from_pretrained(checkpoint)
    return _TOKENIZERS[checkpoint]


def _set_seed(torch, seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_enabled(torch, device: str, amp: bool) -> bool:
    """bf16 только на GPU с его поддержкой (RTX 30xx/40xx — да). Иначе обычный fp32."""
    return bool(amp and device == "cuda" and torch.cuda.is_bf16_supported())


def _autocast(torch, enabled: bool):
    import contextlib
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def class_weights(n_neg: int, n_pos: int, balance: bool) -> list[float]:
    """Веса классов [w0, w1] для cross-entropy: n_total / (2 * n_class). Без балансировки — [1, 1].
    [чистая логика] При 60 vs 14 даёт ≈[0.62, 2.64]: редкий автор весит столько же, сколько частый."""
    if not balance or n_neg == 0 or n_pos == 0:
        return [1.0, 1.0]
    total = n_neg + n_pos
    return [total / (2 * n_neg), total / (2 * n_pos)]


class PairClassifier:
    """Обёртка над дообученным бинарным CAMeLBERT. label 1 = target, 0 = opponent."""

    def __init__(self, model, tokenizer, target: str, opponent: str, device: str = "cpu",
                 train_info: dict | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.target = target
        self.opponent = opponent
        self.device = device
        self.train_info = train_info or {}

    # --- обучение ---
    @classmethod
    def train(
        cls,
        target: str,
        opponent: str,
        train_df,
        checkpoint: str = config.CHECKPOINT,
        epochs: int = config.EPOCHS,
        lr: float = config.LEARNING_RATE,
        batch_size: int = config.BATCH_SIZE,
        max_tokens: int = config.MAX_TOKENS,
        log_every: int = config.LOG_EVERY,
        early_stop_loss: float | None = config.EARLY_STOP_LOSS,
        seed: int = config.SEED,
        amp: bool = config.AMP,
        balance: bool = config.BALANCE_PAIRS,
        max_grad_norm: float | None = config.MAX_GRAD_NORM,
    ) -> "PairClassifier":
        global _DEVICE_REPORTED
        torch, *_ = _lazy()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = _amp_enabled(torch, device, amp)
        if not _DEVICE_REPORTED:
            if device == "cuda":
                log.info("устройство: GPU (%s), обучение в bf16: %s, инференс: fp32",
                         torch.cuda.get_device_name(0), "да" if use_amp else "нет")
            else:
                log.warning("устройство: CPU — torch не видит GPU. Обучение будет в десятки раз медленнее. "
                            "Проверь: python -c \"import torch; print(torch.cuda.is_available())\"")
            _DEVICE_REPORTED = True

        sample_seed = pair_seed(seed, target, opponent) if config.PAIR_SPECIFIC_SAMPLE else seed
        data = pair_frame(train_df, target, opponent, seed=sample_seed)
        if data["label"].nunique() < 2:
            raise ValueError(f"Для пары ({target},{opponent}) нет примеров обоих классов.")

        kw = dict(checkpoint=checkpoint, epochs=epochs, lr=lr, batch_size=batch_size,
                  max_tokens=max_tokens, log_every=log_every, early_stop_loss=early_stop_loss,
                  seed=seed, balance=balance, max_grad_norm=max_grad_norm, device=device)
        try:
            return cls._train_once(target, opponent, data, use_amp=use_amp, **kw)
        except NonFiniteTraining as e:
            if not use_amp:
                raise RuntimeError(f"Пара {target} vs {opponent} не обучилась даже в fp32: {e}") from e
            reason = str(e)
        # Переобучаем ВНЕ блока except: пойманное исключение держит ссылку на кадр неудавшегося
        # обучения (модель + оптимизатор ≈ 1.8 ГБ VRAM), и внутри except в памяти были бы две модели.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.warning("  пара %s vs %s: %s в bf16 -> переобучаю пару с нуля в fp32", target, opponent, reason)
        clf = cls._train_once(target, opponent, data, use_amp=False, **kw)
        clf.train_info["fallback_fp32"] = True
        return clf

    @classmethod
    def _train_once(cls, target, opponent, data, *, use_amp, checkpoint, epochs, lr, batch_size,
                    max_tokens, log_every, early_stop_loss, seed, balance, max_grad_norm,
                    device) -> "PairClassifier":
        torch, DataLoader, _, AutoModelForSeq, AutoTokenizer = _lazy()
        _set_seed(torch, seed)  # воспроизводимость: голова классификатора и порядок батчей

        tok = _get_tokenizer(checkpoint)
        # Без паддинга до 512: паддим каждый батч до самого длинного в нём (динамический паддинг).
        enc = tok(list(data["text"]), truncation=True, max_length=max_tokens)
        lengths = sorted(len(ids) for ids in enc["input_ids"])
        n_cut = sum(1 for n in lengths if n >= max_tokens)
        counts = data["label"].value_counts().to_dict()
        n_pos, n_neg = int(counts.get(1, 0)), int(counts.get(0, 0))
        w = class_weights(n_neg, n_pos, balance)
        log.info("пара %s vs %s: %d vs %d сегм.%s; токенов: медиана %d, макс %d%s; точность обучения: %s",
                 target, opponent, n_pos, n_neg,
                 f" (веса в loss: {target} {w[1]:.2f} / {opponent} {w[0]:.2f})" if w != [1.0, 1.0] else "",
                 lengths[len(lengths) // 2], lengths[-1],
                 f", ОБРЕЗАНО до {max_tokens}: {n_cut}" if n_cut else "",
                 "bf16" if use_amp else "fp32")

        items = [{"input_ids": ids, "attention_mask": am, "label": int(y)}
                 for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], data["label"])]

        def collate(batch):
            padded = tok.pad([{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                              for b in batch], return_tensors="pt")
            padded["labels"] = torch.tensor([b["label"] for b in batch])
            return padded

        loader = DataLoader(items, batch_size=batch_size, shuffle=True, collate_fn=collate)
        weight = torch.tensor(w).to(device)

        model = AutoModelForSeq.from_pretrained(checkpoint, num_labels=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)  # Adam как в статье; lr — см. config.LEARNING_RATE
        t0, low_streak, skipped = time.time(), 0, 0
        loss, acc, ep = float("nan"), 0.0, 0
        for ep in range(1, epochs + 1):
            model.train()
            loss_sum, correct, seen = 0.0, 0, 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                opt.zero_grad()
                with _autocast(torch, use_amp):
                    logits = model(**batch).logits
                step_loss = torch.nn.functional.cross_entropy(logits.float(), labels, weight=weight)
                loss_val = step_loss.item()
                if not math.isfinite(loss_val):
                    if use_amp:
                        raise NonFiniteTraining(f"loss={loss_val} на эпохе {ep}")
                    skipped += 1
                    continue
                step_loss.backward()
                if max_grad_norm:
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
                    if not math.isfinite(gn):
                        if use_amp:
                            raise NonFiniteTraining(f"норма градиента={gn} на эпохе {ep}")
                        skipped += 1
                        opt.zero_grad()
                        continue
                opt.step()
                n = labels.size(0)
                loss_sum += loss_val * n
                correct += (logits.argmax(-1) == labels).sum().item()
                seen += n
            if skipped > config.MAX_SKIPPED_STEPS:
                raise NonFiniteTraining(f"пропущено {skipped} шагов с NaN/inf в fp32")
            if seen == 0:
                raise NonFiniteTraining(f"на эпохе {ep} не осталось ни одного числового шага")
            loss, acc = loss_sum / seen, correct / seen
            elapsed = time.time() - t0
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                log.info("  эпоха %3d/%d  loss=%.4f  train_acc=%.3f  прошло %s  осталось ~%s",
                         ep, epochs, loss, acc, fmt_duration(elapsed),
                         fmt_duration(elapsed / ep * (epochs - ep)))
            if early_stop_loss is not None:
                low_streak = low_streak + 1 if loss < early_stop_loss else 0
                if low_streak >= 3:
                    log.info("  ранняя остановка на эпохе %d: loss < %s три эпохи подряд (%s)",
                             ep, early_stop_loss, fmt_duration(elapsed))
                    break
        if use_amp and not _params_finite(torch, model):
            raise NonFiniteTraining("веса модели стали нечисловыми к концу обучения")
        info = {"target": target, "opponent": opponent, "n_target": n_pos, "n_opponent": n_neg,
                "epochs_run": ep, "final_loss": round(loss, 5), "final_train_acc": round(acc, 4),
                "precision": "bf16" if use_amp else "fp32", "skipped_steps": skipped,
                "fallback_fp32": False, "seconds": round(time.time() - t0, 1)}
        return cls(model, tok, target, opponent, device, train_info=info)

    # --- инференс ---
    def predict_proba(self, text: str) -> float:
        return self.predict_proba_batch([text])[0]

    def predict_proba_batch(self, texts: list[str], batch_size: int = 32) -> list[float]:
        """P(label=1 == target) для списка текстов. Всегда fp32 (дёшево и без риска NaN)."""
        torch, *_ = _lazy()
        self.model.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, truncation=True, padding=True,
                                 max_length=config.MAX_TOKENS, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1)[:, 1]
            out.extend(probs.detach().cpu().tolist())
        out = [float(p) for p in out]
        bad = sum(1 for p in out if not math.isfinite(p))
        if bad:
            raise NonFinitePrediction(f"Пара {self.target} vs {self.opponent}: {bad} нечисловых "
                                      f"вероятностей из {len(out)}")
        return out

    def free(self) -> None:
        """Выгрузить модель из памяти (важно для ансамбля из многих пар)."""
        torch, *_ = _lazy()
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- сериализация ---
    def save(self, model_path: str) -> str:
        p = Path(model_path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        (p / "pair.json").write_text(
            json.dumps({"target": self.target, "opponent": self.opponent,
                        "train_info": self.train_info}, ensure_ascii=False),
            encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, model_path: str) -> "PairClassifier":
        torch, _, _, AutoModelForSeq, AutoTokenizer = _lazy()
        p = Path(model_path)
        meta = json.loads((p / "pair.json").read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq.from_pretrained(p).to(device)
        tok = AutoTokenizer.from_pretrained(p)
        return cls(model, tok, meta["target"], meta["opponent"], device,
                   train_info=meta.get("train_info"))


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(name))[:40]


def train_pair(target: str, opponent: str, train_df, out_dir=None, **cfg) -> str:
    """Обучает одну пару и сохраняет на диск. Возвращает путь к модели."""
    out_dir = Path(out_dir or config.MODELS_DIR) / f"{_slug(target)}__vs__{_slug(opponent)}"
    clf = PairClassifier.train(target, opponent, train_df, **cfg)
    return clf.save(out_dir)
