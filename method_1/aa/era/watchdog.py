"""Сторож зависаний для долгих прогонов без присмотра.

Обучение и предсказание вызывают beat() на каждом шаге. Если beat() не было дольше timeout,
сторож пишет в лог стек всех потоков (где именно застряло) и завершает процесс с кодом 3.
Внешний цикл (см. aa/era/README.md, «Запуск без присмотра») перезапускает прогон с --resume,
и уже посчитанные фолды / seed не пересчитываются.

Зачем выход, а не «разбудить»: зависание внутри CUDA-вызова из Python не прервать.
"""
from __future__ import annotations

import faulthandler
import logging
import os
import threading
import time

log = logging.getLogger("aa.era.watchdog")
_last = time.monotonic()
_what = "старт"
_started = False


def beat(what: str | None = None) -> None:
    global _last, _what
    _last = time.monotonic()
    if what:
        _what = what


def start(timeout_min: float) -> None:
    """Запустить сторож (один раз на процесс). timeout_min <= 0 — выключен."""
    global _started
    if _started or timeout_min <= 0:
        return
    _started = True
    beat("старт")

    def run():
        while True:
            time.sleep(30)
            idle = time.monotonic() - _last
            if idle > timeout_min * 60:
                log.critical("СТОРОЖ: %.0f мин без прогресса (последнее: %s) — пишу стек и завершаю процесс; "
                             "перезапусти с --resume (или используй цикл из README)", idle / 60, _what)
                for h in logging.getLogger("aa").handlers:
                    stream = getattr(h, "stream", None)
                    if stream is not None:
                        try:
                            faulthandler.dump_traceback(file=stream, all_threads=True)
                            stream.flush()
                        except Exception:  # noqa: BLE001
                            pass
                os._exit(3)

    threading.Thread(target=run, name="era-watchdog", daemon=True).start()
    log.info("сторож зависаний включён: %.0f мин без прогресса -> выход с кодом 3", timeout_min)
