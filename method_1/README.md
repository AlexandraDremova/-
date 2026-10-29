# poetry-aa — атрибуция авторства доисламских арабских поэтов

Воспроизведение метода из *BERT-based Classical Arabic Poetry Authorship Attribution*
(COLING 2025, Alqurashi et al.): ансамбль попарных CAMeLBERT-классификаторов + majority vote,
с оценкой уверенности для детекции неверной атрибуции.

## Две дорожки
- **Данные и оценка** (`aa/data`, `aa/eval`) — Элина
- **Модель и инференс** (`aa/model`) — сокомандник

Граница между ними — два контракта (см. `aa/model/interface.py` и `aa/schema.py`).
Пока реальной модели нет, eval работает против `aa/model/stub.py`, поэтому обе дорожки
стартуют независимо.

## Установка
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# camel-tools и torch/transformers тяжёлые — для дорожки данных/оценки они не обязательны:
# препроцессинг и eval работают и без них (есть fallback и стаб-модель).
```

## Проверка, что всё живо (end-to-end на игрушечных данных)
```bash
python -m aa.data.make_toy            # создаёт data/toy/dataset.parquet
pytest -q                             # прогон стаб-модели сквозь eval-харнесс
```
Если `pytest` зелёный — труба собрана: данные → (стаб) обучение → предсказание с confidence → F1.

## Структура
```
aa/
  config.py          общие параметры (чекпоинт, сегментация, обучение)
  schema.py          КОНТРАКТ ДАННЫХ: колонки одной строки датасета
  data/
    preprocess.py    чистка текста (диакритика, кашида) — с fallback без camel-tools
    segment.py       нарезка поэм на окна бейтов под лимит 512 токенов
    build_dataset.py сборка mualakat.json + diwans.zip + APCD -> parquet  [твоя работа]
    make_toy.py      игрушечный датасет для тестов
  model/
    interface.py     КОНТРАКТ МОДЕЛИ: сигнатуры fit / train_pair / predict
    stub.py          заглушка — чтобы eval запускался без реальной модели
  eval/
    harness.py       кросс-валидация, агрег'0ация F1, таблица предсказаний  [твоя работа]
    calibrate.py     порог уверенности из распределений confidence          [твоя работа]
  era/               классификатор ЭПОХИ на CAMeLBERT по APCD (CV по поэтам, P(доисламская) муаллак)
                     — см. aa/era/README.md
tests/
  test_end_to_end.py
  test_era.py
```

## Классификатор эпохи (APCD)
```bash
python -m aa.era.apcd --apcd arbml/APCD --project-data data/processed/dataset.parquet
python -m aa.era.run_era_cv --scheme coarse        # CV по поэтам + базовые линии
python -m aa.era.run_era_cases --scheme coarse     # муаллаки против эталона из диванов
```
