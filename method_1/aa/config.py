"""Общие параметры проекта. Меняется здесь, а не по месту в коде."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"          # сюда кладём mualakat.json, распакованные diwans, APCD
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_TOY = ROOT / "data" / "toy"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

# --- Модель (дорожка сокомандника; тут только дефолты, которые видит eval) ---
CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-ca"   # альтернатива для экспа: ...camelbert-mix
MAX_TOKENS = 512

# --- Сегментация (дорожка данных) ---
SEGMENT_WINDOW = 8     # бейтов (строк) в одном сегменте
SEGMENT_STRIDE = 8     # шаг окна; < window => перекрытие
MIN_SEGMENT_BAYTS = 3  # сегменты короче — отбрасываем

# --- Дубли и утечки ---
FUZZY_DUP_THRESHOLD = 0.5  # Жаккар символьных триграмм: бейт с J ≥ 0.5 = вариант бейта муаллаки
                           # (≈ отличие в 1–2 слова) -> сегмент убирается из обучения; и та же
                           # мера объединяет поэмы в группы для кросс-валидации

# --- Обучение / ансамбль (дефолты из статьи, перебираются в экспериментах) ---
EXAMPLES_PER_POET = 60
OPPONENTS_PER_TARGET = 1
EPOCHS = 100
N_SPLITS = 5
LEARNING_RATE = 2e-5       # ОТСТУПЛЕНИЕ от статьи: там "Adam с lr по умолчанию" (в PyTorch/Keras это 1e-3).
                           # 1e-3 для дообучения BERT обычно разваливает обучение; 2e-5 — стандарт.
BATCH_SIZE = 16
MAX_OPPONENTS = None       # None => полный round-robin (каждая модель = 1 оппонент, как оптимум статьи)
                           # int => подвыборка оппонентов на поэта (дешевле, тот же knob из статьи)
SEED = 42
AMP = True                 # bf16 autocast при ОБУЧЕНИИ на GPU (~3x быстрее fp32); инференс всегда fp32.
                           # При NaN/inf в bf16 пара автоматически переобучается в fp32.
MAX_GRAD_NORM = 1.0        # клиппинг нормы градиента (стандарт для дообучения BERT); None = выкл.
MAX_SKIPPED_STEPS = 3      # в fp32: сколько шагов с NaN/inf можно пропустить, прежде чем считать пару сломанной
PAIR_SPECIFIC_SAMPLE = True  # своя случайная выборка 60 сегментов для каждой пары (иначе все пары
                             # с крупным поэтом видят одни и те же 60 сегментов из 120+)
BALANCE_PAIRS = True       # взвешенный loss, если в паре разное число сегментов (напр. 60 vs 14).
                           # В статье пары сбалансированы (60/60); у нас у малых диванов меньше 60.
LOG_EVERY = 5             # писать строку прогресса каждые N эпох (и всегда первую/последнюю)
EARLY_STOP_LOSS = None     # напр. 0.01: остановить пару, когда train-loss ниже порога 3 эпохи подряд.
                           # None = честные EPOCHS как в статье. Сильно экономит время (модель
                           # запоминает ~120 примеров за 10-20 эпох, дальше эпохи почти пустые).


# --- Классификатор эпохи (aa/era) ---
ERA_SCHEME = "coarse"          # binary | coarse | full — см. aa/era/labels.py
ERA_EPOCHS = 4                 # многоклассовая модель на тысячах сегментов: 3–4 эпох обычно хватает
ERA_BATCH_SIZE = 32
ERA_LEARNING_RATE = 2e-5
ERA_WARMUP = 0.06              # доля шагов линейного разогрева lr
ERA_WEIGHT_DECAY = 0.01
ERA_PATIENCE = 2               # ранняя остановка: столько эпох без роста macro-F1 на валидации
ERA_MAX_SEG_PER_POET = 40      # обучение: не больше N сегментов от одного поэта (плодовитые не перетягивают)
ERA_MAX_SEG_PER_CLASS = 6000   # обучение: не больше N сегментов на класс (время на RTX 4060)
ERA_EVAL_SEG_PER_POET = 20     # тест/валидация: не больше N сегментов от поэта
ERA_EVAL_SEG_PER_CLASS = 3000
ERA_MIN_POETS_PER_CLASS = 8    # класс с меньшим числом поэтов выбрасывается (его нельзя честно проверить)
ERA_ATTN = "eager"             # реализация внимания BERT. "eager" — классическая; "sdpa" (умолчание transformers)
                               # давала NaN в батчах с паддингом (прогоны 28–29.10), см. aa/era/README.md
ERA_FUZZY_ERAS = ("pre_islamic", "mukhadram", "islamic", "umayyad")  # где искать варианты бейтов проекта


@dataclass
class Periods:
    """Границы периодов из статьи — для фильтрации и контрастного пула."""
    pre_islamic: tuple[int, int] = (-600, -570)  # условно; реальная разметка идёт по веку/дате смерти
    labels: list[str] = field(default_factory=lambda: [
        "pre-islamic", "veteran", "islamic", "umayyad", "abbasid",
        "andalusian", "ayyubi", "mamluk", "ottoman",
    ])


for _d in (DATA_RAW, DATA_PROCESSED, DATA_TOY, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
