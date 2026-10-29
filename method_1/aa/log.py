"""Единая настройка логирования: консоль + файл logs/<имя>_<время>.log.

Скрипты (run_cv, run_cases, diagnose_pair) вызывают setup_logging() в начале.
Модули пишут через logging.getLogger(__name__) — все они дети логгера "aa".
Долгие прогоны можно смотреть из другого терминала: tail -f logs/<файл>.log
"""
from __future__ import annotations

import faulthandler
import logging
import sys
from datetime import datetime
from pathlib import Path

from aa import config

_CONFIGURED = False
_FAULT_FILE = None


def _log_uncaught(exc_type, exc, tb):
    """Необработанное исключение -> в лог-файл с трейсбеком (раньше оно было только в терминале,
    и по логу нельзя было понять, почему прогон оборвался)."""
    log = logging.getLogger("aa")
    if issubclass(exc_type, KeyboardInterrupt):
        log.error("прогон прерван вручную (Ctrl+C)")
    else:
        log.critical("прогон упал с необработанным исключением:", exc_info=(exc_type, exc, tb))
    sys.__excepthook__(exc_type, exc, tb)


def setup_logging(name: str = "run", to_file: bool = True, level: int = logging.INFO) -> Path | None:
    global _CONFIGURED
    root = logging.getLogger("aa")
    if _CONFIGURED:
        return None
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    path = None
    if to_file:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.LOGS_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                          "%Y-%m-%d %H:%M:%S"))
        root.addHandler(fh)

    sys.excepthook = _log_uncaught
    if path:
        # падения на уровне C (segfault, ошибки драйвера) — тоже в лог-файл
        global _FAULT_FILE
        _FAULT_FILE = open(path, "a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_FILE)
        try:   # зависание: `kill -USR1 <pid>` пишет в лог, где сейчас стоит каждый поток
            import signal
            faulthandler.register(signal.SIGUSR1, file=_FAULT_FILE, all_threads=True)
        except (AttributeError, ValueError):  # Windows без SIGUSR1
            pass

    _CONFIGURED = True
    if path:
        root.info("лог пишется в %s", path)
    return path


def mem_usage() -> str:
    """Текущая память процесса и видеопамять — для диагностики падений по нехватке памяти.
    (Если процесс убьёт OOM-killer, трейсбека не будет; рост памяти в логе это покажет.)"""
    parts = []
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    kb = int(line.split()[1])
                    parts.append(("RAM" if line.startswith("VmRSS") else "пик RAM") + f" {kb / 1048576:.1f} ГБ")
    except OSError:
        pass
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None:
        try:
            if torch.cuda.is_available():
                parts.append(f"VRAM {torch.cuda.memory_allocated() / 2**30:.1f} ГБ "
                             f"(резерв {torch.cuda.memory_reserved() / 2**30:.1f})")
        except Exception:  # noqa: BLE001 — диагностика не должна ронять прогон
            pass
    return ", ".join(parts)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("aa") else f"aa.{name}")


def fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m:02d}м"
    if m:
        return f"{m}м {s:02d}с"
    return f"{s}с"
