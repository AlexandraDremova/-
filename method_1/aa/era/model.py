"""Многоклассовый CAMeLBERT для эпохи.  [torch/transformers — ленивый импорт]

Одна модель BertForSequenceClassification на C классов схемы (а не 45 попарных, как для авторов:
классов мало, данных по каждому — тысячи сегментов).

Обучение:
  * AdamW, lr 2e-5, линейный разогрев (ERA_WARMUP) и затухание, weight decay (кроме bias/LayerNorm);
  * взвешенный cross-entropy (классы несбалансированы даже после ограничения выборки);
  * bf16 на GPU; при NaN/inf модель переобучается с нуля в fp32 (как пары в finetune.py);
  * клиппинг нормы градиента;
  * после каждой эпохи — балансная точность на ВАЛИДАЦИОННЫХ ПОЭТАХ; лучшая эпоха сохраняется,
    ранняя остановка после ERA_PATIENCE эпох без улучшения;
  * прогресс и оставшееся время пишутся в лог каждые log_every шагов.
Инференс всегда в fp32, тексты сортируются по длине (меньше паддинга).

    clf = EraClassifier.train(train_df, val_df, classes)      # колонки text, label
    probs = clf.predict_proba(texts)                           # (n, C), порядок = clf.classes
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from aa import config
from aa.era import watchdog
from aa.log import fmt_duration, mem_usage
from aa.model.finetune import (NonFiniteTraining, _amp_enabled, _autocast,
                               _get_tokenizer, _lazy, _params_finite, _set_seed)

log = logging.getLogger("aa.era.model")


def _balanced_acc(y_true, y_pred, n_classes: int) -> float:
    from aa.era.metrics import confusion, from_confusion
    return from_confusion(confusion(y_true, y_pred, n_classes), "balanced_acc")


class EraClassifier:
    def __init__(self, model, tokenizer, classes: list[str], device: str = "cpu",
                 train_info: dict | None = None, max_tokens: int = config.MAX_TOKENS):
        self.model = model
        self.tokenizer = tokenizer
        self.classes = list(classes)
        self.device = device
        self.train_info = train_info or {}
        self.max_tokens = max_tokens

    # ---------- обучение ----------
    @classmethod
    def train(cls, train_df, val_df, classes: list[str],
              checkpoint: str = config.CHECKPOINT,
              epochs: int = config.ERA_EPOCHS,
              lr: float = config.ERA_LEARNING_RATE,
              batch_size: int = config.ERA_BATCH_SIZE,
              warmup: float = config.ERA_WARMUP,
              weight_decay: float = config.ERA_WEIGHT_DECAY,
              patience: int | None = config.ERA_PATIENCE,
              class_weights: list[float] | None = None,
              max_tokens: int = config.MAX_TOKENS,
              seed: int = config.SEED,
              amp: bool = config.AMP,
              max_grad_norm: float | None = config.MAX_GRAD_NORM,
              attn: str | None = config.ERA_ATTN,
              log_every: int = 100) -> "EraClassifier":
        torch, *_ = _lazy()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        use_amp = _amp_enabled(torch, device, amp)
        if device == "cuda":
            log.info("устройство: GPU (%s), обучение в bf16: %s", torch.cuda.get_device_name(0),
                     "да" if use_amp else "нет")
        else:
            log.warning("устройство: CPU — torch не видит GPU, обучение будет очень долгим")
        kw = dict(checkpoint=checkpoint, epochs=epochs, lr=lr, batch_size=batch_size, warmup=warmup,
                  weight_decay=weight_decay, patience=patience, class_weights=class_weights,
                  max_tokens=max_tokens, seed=seed, max_grad_norm=max_grad_norm, log_every=log_every,
                  device=device)
        # Лестница попыток: редкие NaN внутри попытки просто пропускаются (см. _train_once);
        # на следующую ступень переходим, только если модель действительно сломалась.
        # bf16 + eager быстрее любой попытки в fp32; fp32 — последняя ступень.
        attempts = list(dict.fromkeys([(use_amp, attn), (use_amp, "eager"), (False, "eager")]))
        last = None
        for n, (amp_i, attn_i) in enumerate(attempts):
            if n:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.warning("попытка %d/%d: %s -> переобучаю с нуля (%s, внимание: %s)", n + 1, len(attempts),
                            last, "bf16" if amp_i else "fp32", attn_i or "по умолчанию")
            try:
                clf = cls._train_once(train_df, val_df, classes, use_amp=amp_i, attn=attn_i, **kw)
                clf.train_info["attempt"] = n + 1
                clf.train_info["fallback_fp32"] = bool(use_amp and not amp_i)
                return clf
            except NonFiniteTraining as e:
                last = str(e)
        raise NonFiniteTraining(f"модель не обучилась ни в одной из {len(attempts)} попыток: {last}")

    @classmethod
    def _train_once(cls, train_df, val_df, classes, *, use_amp, checkpoint, epochs, lr, batch_size,
                    warmup, weight_decay, patience, class_weights, max_tokens, seed, max_grad_norm,
                    log_every, attn, device) -> "EraClassifier":
        torch, DataLoader, _, AutoModelForSeq, _ = _lazy()
        from transformers import get_linear_schedule_with_warmup
        _set_seed(torch, seed)
        if device == "cuda" and not use_amp:
            # TF32 для матричных умножений в fp32: на RTX 30xx/40xx в 2–3 раза быстрее, диапазон как у fp32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        cidx = {c: i for i, c in enumerate(classes)}
        tok = _get_tokenizer(checkpoint)
        y = np.array([cidx[c] for c in train_df["label"]])
        enc = tok(list(train_df["text"]), truncation=True, max_length=max_tokens)
        lengths = sorted(len(ids) for ids in enc["input_ids"])
        n_cut = sum(1 for n in lengths if n >= max_tokens)
        log.info("обучение: %d сегментов, %d поэтов; по классам: %s; токенов медиана %d, макс %d%s; %s",
                 len(train_df), train_df["author"].nunique(),
                 {c: int((y == i).sum()) for c, i in cidx.items()},
                 lengths[len(lengths) // 2], lengths[-1],
                 f", обрезано до {max_tokens}: {n_cut}" if n_cut else "", "bf16" if use_amp else "fp32")

        items = [{"input_ids": a, "attention_mask": m, "label": int(t), "idx": i}
                 for i, (a, m, t) in enumerate(zip(enc["input_ids"], enc["attention_mask"], y))]
        n_tok = np.array([len(a) for a in enc["input_ids"]])

        def collate(batch):
            padded = tok.pad([{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}
                              for b in batch], return_tensors="pt")
            padded["labels"] = torch.tensor([b["label"] for b in batch])
            padded["idx"] = torch.tensor([b["idx"] for b in batch])
            return padded

        gen = torch.Generator()
        gen.manual_seed(seed)
        loader = DataLoader(items, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=gen)

        extra = {"attn_implementation": attn} if attn else {}
        model = AutoModelForSeq.from_pretrained(
            checkpoint, num_labels=len(classes),
            id2label=dict(enumerate(classes)), label2id=cidx, **extra).to(device)
        log.info("попытка: %s, внимание: %s", "bf16" if use_amp else "fp32",
                 getattr(model.config, "_attn_implementation", "?"))
        no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=lr)
        total_steps = max(1, epochs * len(loader))
        sched = get_linear_schedule_with_warmup(opt, int(warmup * total_steps), total_steps)
        weight = torch.tensor(class_weights or [1.0] * len(classes), dtype=torch.float32).to(device)

        clf = cls(model, tok, classes, device, max_tokens=max_tokens)
        has_val = val_df is not None and len(val_df) > 0
        y_val = np.array([cidx[c] for c in val_df["label"]]) if has_val else None

        best, best_state, best_ep, bad, history = -1.0, None, 0, 0, []
        t0, step, skipped, ep = time.time(), 0, 0, 0
        # Порог «модель сломалась» (тогда — следующая попытка); ниже порога NaN просто пропускаются.
        max_skipped = max(10, int(0.10 * total_steps))
        max_bad = max(20, int(0.05 * len(train_df)))
        bad_rows: dict[int, dict] = {}       # примеры с нечисловым loss: idx -> диагностика
        grad_events: list[dict] = []         # шаги с нечисловой нормой градиента
        snap_every, max_restores, restores = 100, 3, 0
        snapshot = cls._snapshot(torch, model, step=0)

        def recover(where: str) -> None:
            """Веса испорчены -> откат к последнему снимку (модель + оптимизатор); после max_restores — сдаёмся."""
            nonlocal restores
            if restores >= max_restores:
                cls._dump_bad(train_df, n_tok, bad_rows, grad_events)
                raise NonFiniteTraining(f"веса стали нечисловыми ({where}) уже {restores + 1}-й раз")
            restores += 1
            model.load_state_dict(snapshot["model"])
            opt.state.clear()             # моменты Adam тоже могли испортиться — начинаем их заново
            opt.zero_grad()
            log.warning("  %s: веса стали нечисловыми -> откат к снимку шага %d (откат %d/%d), обучение продолжается",
                        where, snapshot["step"], restores, max_restores)
        for ep in range(1, epochs + 1):
            model.train()
            loss_sum, correct, seen = 0.0, 0, 0
            for batch in loader:
                watchdog.beat(f"обучение, эпоха {ep}, шаг {step + 1}")
                ids = batch.pop("idx")
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                opt.zero_grad()
                with _autocast(torch, use_amp):
                    logits = model(**batch).logits.float()
                # loss по каждому примеру: так видно, КАКИЕ примеры дают NaN, и их можно выкинуть из шага
                per = torch.nn.functional.cross_entropy(logits, labels, weight=weight, reduction="none")
                ok = torch.isfinite(per)
                step += 1
                if not bool(ok.all()):
                    bad_ids = ids[~ok.cpu()].tolist()
                    for i in bad_ids:
                        if i not in bad_rows:
                            bad_rows[i] = {"step": step, "batch_max_tokens": int(n_tok[ids.numpy()].max()),
                                           "alone_finite": cls._finite_alone(torch, model, tok, enc, i, device)}
                    log.warning("  шаг %d: нечисловой loss у %d из %d примеров (idx %s; токенов %s; "
                                "отдельно, без паддинга, число: %s) — выкидываю их из шага",
                                step, len(bad_ids), len(ids), bad_ids[:5], n_tok[bad_ids[:5]].tolist(),
                                [bad_rows[i]["alone_finite"] for i in bad_ids[:5]])
                    if not _params_finite(torch, model):
                        recover(f"шаг {step}")
                        sched.step()
                        continue
                    if len(bad_rows) > max_bad:
                        cls._dump_bad(train_df, n_tok, bad_rows, grad_events)
                        raise NonFiniteTraining(f"нечисловой loss у {len(bad_rows)} обучающих примеров "
                                                f"(> {max_bad}); список — в logs/era_nonfinite_*.csv")
                    if not bool(ok.any()):
                        skipped += 1
                        sched.step()
                        continue
                loss = per[ok].sum() / weight[labels[ok]].sum()
                lv = loss.item()
                loss.backward()
                if max_grad_norm:
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm))
                    if not math.isfinite(gn):
                        skipped += 1
                        grad_events.append({"step": step, "loss": lv, "idx": ids.tolist(),
                                            "batch_max_tokens": int(n_tok[ids.numpy()].max())})
                        log.warning("  шаг %d: норма градиента %s при loss=%.4f (батч %d примеров, до %d токенов) "
                                    "— шаг пропущен", step, gn, lv, len(ids), int(n_tok[ids.numpy()].max()))
                        opt.zero_grad()
                        sched.step()
                        if skipped > max_skipped:
                            cls._dump_bad(train_df, n_tok, bad_rows, grad_events)
                            raise NonFiniteTraining(f"пропущено {skipped} шагов (> {max_skipped}) с нечисловой "
                                                    f"нормой градиента; детали — logs/era_nonfinite_*.csv")
                        if not _params_finite(torch, model):
                            recover(f"шаг {step}")
                        continue
                opt.step()
                sched.step()
                if step % snap_every == 0:
                    if _params_finite(torch, model):
                        snapshot = cls._snapshot(torch, model, step, into=snapshot)
                    else:
                        recover(f"шаг {step}")
                        continue
                labels, logits = labels[ok], logits[ok]
                n = labels.size(0)
                loss_sum += lv * n
                correct += (logits.argmax(-1) == labels).sum().item()
                seen += n
                if step % log_every == 0:
                    el = time.time() - t0
                    log.info("  эпоха %d/%d, шаг %d/%d  loss=%.4f  train_acc=%.3f  прошло %s  осталось ~%s  (%s)",
                             ep, epochs, step, total_steps, loss_sum / max(seen, 1), correct / max(seen, 1),
                             fmt_duration(el), fmt_duration(el / step * (total_steps - step)), mem_usage())
            if seen == 0:
                raise NonFiniteTraining(f"на эпохе {ep} не осталось ни одного числового шага")
            if not _params_finite(torch, model):
                recover(f"конец эпохи {ep}")

            rec = {"epoch": ep, "train_loss": round(loss_sum / seen, 5), "train_acc": round(correct / seen, 4)}
            if has_val:
                pv = clf.predict_proba(list(val_df["text"])).argmax(axis=1)
                rec["val_balanced_acc"] = round(_balanced_acc(y_val, pv, len(classes)), 4)
                model.train()
            history.append(rec)
            log.info("эпоха %d/%d: %s  (%s; %s)", ep, epochs, rec, fmt_duration(time.time() - t0), mem_usage())

            if has_val:
                if rec["val_balanced_acc"] > best:
                    best, best_ep, bad = rec["val_balanced_acc"], ep, 0
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if patience is not None and bad >= patience:
                        log.info("ранняя остановка: %d эпох без роста балансной точности на валидации", bad)
                        break

        if bad_rows or grad_events:
            cls._dump_bad(train_df, n_tok, bad_rows, grad_events)
        if best_state is not None and best_ep != ep:
            model.load_state_dict(best_state)
            log.info("возвращаю веса лучшей эпохи %d (val balanced_acc %.4f)", best_ep, best)
        clf.train_info = {"epochs_run": ep, "best_epoch": best_ep or ep,
                          "best_val_balanced_acc": best if has_val else None,
                          "precision": "bf16" if use_amp else "fp32", "fallback_fp32": False,
                          "skipped_steps": skipped, "nonfinite_examples": len(bad_rows), "restores": restores,
                          "attn": getattr(model.config, "_attn_implementation", None),
                          "seconds": round(time.time() - t0, 1),
                          "n_train": len(train_df), "n_val": len(val_df) if has_val else 0,
                          "history": history, "checkpoint": checkpoint, "seed": seed}
        return clf

    @staticmethod
    def _snapshot(torch, model, step: int, into: dict | None = None) -> dict:
        """Копия ВЕСОВ в оперативной памяти (~0.45 ГБ). Буферы выделяются один раз и дальше
        перезаписываются на месте — память не растёт и не удваивается на время копирования
        (прежний снимок с состоянием Adam занимал ~1.3 ГБ и каждый раз выделялся заново)."""
        sd = model.state_dict()
        if into is None:
            return {"step": step, "model": {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}}
        with torch.no_grad():
            for k, v in sd.items():
                into["model"][k].copy_(v.detach())
        into["step"] = step
        return into

    @staticmethod
    def _finite_alone(torch, model, tok, enc, i: int, device) -> bool:
        """Тот же пример ОДИН, без паддинга, в fp32: число ли loss? Отличает «плохой текст» от
        «плохого батча» (паддинг/внимание)."""
        was_training = model.training
        model.eval()
        with torch.no_grad():
            x = tok.pad([{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}],
                        return_tensors="pt").to(device)
            ok = bool(torch.isfinite(model(**x).logits.float()).all())
        if was_training:
            model.train()
        return ok

    @staticmethod
    def _dump_bad(train_df, n_tok, bad_rows: dict, grad_events: list) -> None:
        from datetime import datetime
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.LOGS_DIR / f"era_nonfinite_{datetime.now():%Y%m%d_%H%M%S}.csv"
        rows = []
        for i, d in bad_rows.items():
            r = train_df.iloc[i]
            rows.append({"тип": "loss", "idx": i, "шаг": d["step"], "токенов": int(n_tok[i]),
                         "макс_токенов_в_батче": d["batch_max_tokens"], "один_без_паддинга_число": d["alone_finite"],
                         "poem_id": r.get("poem_id"), "author": r.get("author"), "label": r.get("label"),
                         "text": r.get("text")})
        for g in grad_events:
            rows.append({"тип": "градиент", "idx": " ".join(map(str, g["idx"])), "шаг": g["step"],
                         "макс_токенов_в_батче": g["batch_max_tokens"]})
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
        log.warning("примеры и шаги с NaN/inf записаны в %s", path)

    # ---------- инференс ----------
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        torch, *_ = _lazy()
        self.model.eval()
        texts = list(texts)
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), len(self.classes)), dtype=np.float64)
        for s in range(0, len(texts), batch_size):
            watchdog.beat(f"предсказание, {s}/{len(texts)}")
            idx = order[s:s + batch_size]
            enc = self.tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                                 max_length=self.max_tokens, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            out[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
        bad = np.flatnonzero(~np.isfinite(out).all(axis=1))
        if bad.size:      # повтор по одному, без паддинга
            for i in bad:
                enc = self.tokenizer([texts[i]], truncation=True, max_length=self.max_tokens,
                                     return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out[i] = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()[0]
            still = np.flatnonzero(~np.isfinite(out).all(axis=1))
            log.warning("предсказание: %d текстов дали NaN в батче, по одному без паддинга — %d из них всё ещё NaN",
                        bad.size, still.size)
            if still.size:   # не падаем: строки остаются NaN, вызывающий код их исключает
                log.warning("  тексты с NaN (исключаются из оценки): %s", [texts[i][:60] for i in still[:3]])
        return out

    def free(self) -> None:
        torch, *_ = _lazy()
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- сериализация ----------
    def save(self, path) -> str:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        (p / "era.json").write_text(json.dumps({"classes": self.classes, "train_info": self.train_info,
                                                "max_tokens": self.max_tokens}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path) -> "EraClassifier":
        torch, _, _, AutoModelForSeq, AutoTokenizer = _lazy()
        p = Path(path)
        meta = json.loads((p / "era.json").read_text(encoding="utf-8"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq.from_pretrained(p).to(device)
        return cls(model, AutoTokenizer.from_pretrained(p), meta["classes"], device,
                   train_info=meta.get("train_info"), max_tokens=meta.get("max_tokens", config.MAX_TOKENS))


def bert_fit_predict(train_df, val_df, classes, predict_texts: dict[str, list[str]], save_to=None,
                     **cfg) -> tuple[dict[str, np.ndarray], dict]:
    """Обучить, предсказать несколько наборов текстов, выгрузить модель.
    Возвращает ({имя набора: вероятности}, train_info). Тот же контракт у заглушки в тестах."""
    clf = EraClassifier.train(train_df, val_df, classes, **cfg)
    out = {name: clf.predict_proba(texts) for name, texts in predict_texts.items()}
    if save_to:
        clf.save(save_to)
        log.info("модель сохранена: %s", save_to)
    info = clf.train_info
    clf.free()
    return out, info
