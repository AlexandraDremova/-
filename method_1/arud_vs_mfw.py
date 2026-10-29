import glob
import json
import re
from collections import Counter
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import rand_score, adjusted_rand_score

SHORT_VOWELS = {'\u064E', '\u064F', '\u0650'}
TANWEEN = {'\u064B', '\u064C', '\u064D'}
SUKUN = '\u0652'
SHADDA = '\u0651'
DAGGER_ALIF = '\u0670'
ALL_DIACRITICS = SHORT_VOWELS | TANWEEN | {SUKUN, SHADDA, DAGGER_ALIF, '\u0640'}

LONG_VOWELS = {'ا', 'و', 'ي', 'ى'}
ALEF_VARIANTS = {'أ', 'إ', 'آ', 'ٱ', 'ا'}


def normalize_for_mfw(word: str) -> str:
    word = re.sub(r'[^\u0621-\u063A\u0641-\u064A]', '', word)
    word = re.sub(r'[أإآٱ]', 'ا', word)
    return word.strip()


def encode_word_arud_strict(word: str) -> str:
    """
    Кодирование Хашана (BASRAH) с фонотактической защитой от пропущенных харакатов:
    - Запрещает невозможные в Аруде цепочки сукунов ('00').
    - Корректно разбирает слитные префиксы с артиклем (فال-, وال-, بال-, كال-, لل-).
    """
    w = re.sub(r'[^\u0621-\u063A\u0641-\u0652\u0670]', '', word)
    if not w:
        return ""

    clusters = []
    for ch in w:
        if ch in ALL_DIACRITICS:
            if clusters:
                clusters[-1][1].add(ch)
        else:
            clusters.append([ch, set()])

    if not clusters:
        return ""

    letters = [c[0] for c in clusters]
    n = len(clusters)

    # Определяем индексы непроизносимых букв (хамзату-ль-васль после префиксов и солнечный лам)
    skip_indices = set()
    # Начальный артикль ال
    if n >= 2 and letters[0] in ALEF_VARIANTS and letters[1] == 'ل':
        if n >= 3 and SHADDA in clusters[2][1]:
            skip_indices.add(1)  # солнечный лам не читается
    # Префикс (و, ف, ب, ك) + артикль ال (например, فاليوم)
    if n >= 3 and letters[0] in ('و', 'ف', 'ب', 'ك') and letters[1] in ALEF_VARIANTS and letters[2] == 'ل':
        skip_indices.add(1)      # алиф аль-васль после префикса никогда не читается
        if n >= 4 and SHADDA in clusters[3][1]:
            skip_indices.add(2)  # солнечный лам не читается

    code = []

    def append_bit(bit_str: str):
        """Добавляет биты с соблюдением запрета на '00' (ильтика ас-сакинейн)."""
        for b in bit_str:
            if b == '0' and (not code or code[-1] == '0'):
                # В начале слова или сразу после сукуна '0' стоять не может -> это согласный с пропущенной гласной
                code.append('1')
            else:
                code.append(b)

    for i, (char, diacs) in enumerate(clusters):
        if i in skip_indices:
            continue

        # 1. Алиф-мадда (آ) -> 10
        if char == 'آ':
            append_bit('10')
            continue

        # 2. Непроизносимый алиф после танвин-фатхи в конце слова (كتاباً)
        if i == n - 1 and char in ('ا', 'ى') and i > 0 and ('\u064B' in clusters[i - 1][1]):
            continue

        # 3. Непроизносимый алиф аль-вакайа после окончания глагола وا (قالوا)
        if i == n - 1 and char == 'ا' and i > 0 and letters[i - 1] == 'و' and not diacs:
            continue

        has_shadda = SHADDA in diacs
        has_short = bool(diacs & SHORT_VOWELS)
        has_tanween = bool(diacs & TANWEEN)
        has_sukun = SUKUN in diacs
        has_dagger = DAGGER_ALIF in diacs

        # 4. Шадда = сукун (0) + огласованный согласный (1)
        if has_shadda:
            if has_tanween:
                append_bit('010')
            elif has_sukun:
                append_bit('00')
            else:
                append_bit('01')
            continue

        # 5. Танвин = краткая гласная + нун с сукуном -> 10
        if has_tanween:
            append_bit('10')
            continue

        # 6. Явная краткая гласная
        if has_short:
            append_bit('1')
            if has_dagger:
                append_bit('0')
            continue

        # 7. Явный сукун
        if has_sukun:
            append_bit('0')
            continue

        # 8. Буква без диакритики:
        # Если это артикль ال (лам без шадды на следующей букве) — на ламе всегда сукун (0)
        if char == 'ل' and ((i == 1 and letters[0] in ALEF_VARIANTS) or (i == 2 and 1 in skip_indices)):
            append_bit('0')
            continue

        # Если это долгая гласная (ا, و, ي, ى):
        if char in LONG_VOWELS or char == 'ٱ':
            if i == 0:
                append_bit('1')
            else:
                # Если перед و или ي уже стоит 0 (например, сукун на ламе в فاليوم),
                # то append_bit('0') автоматически превратит её в '1' (согласный йа/вав с гласной)!
                append_bit('0')
        else:
            # Согласная без огласовки:
            # Типичные служебные слова с сукуном на конце (من, عن, قد, لم, لن, بل, هل, كم, إذ, إن)
            if i == n - 1 and n == 2 and letters[0] in ('م', 'ع', 'ق', 'ل', 'ب', 'ه', 'ك', 'إ', 'أ') and char in ('ن', 'د', 'م', 'ل', 'ذ'):
                append_bit('0')
            else:
                # Во всех остальных случаях отсутствие значка на согласной — это просто лень наборщика, а не сукун!
                append_bit('1')

    return "".join(code)


def load_corpus(raw_dir="data/raw/diwans"):
    author_verses = {}
    for f in sorted(glob.glob(f"{raw_dir}/*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            poems = json.load(fp)
        for p in poems:
            poet = p.get("poet_slug") or p.get("poet")
            verses = p.get("verses", [])
            if poet not in author_verses:
                author_verses[poet] = []
            for v in verses:
                line = " ".join(v) if isinstance(v, list) else str(v)
                if line.strip():
                    author_verses[poet].append(line.strip())
    return author_verses


def sample_non_overlapping_chunks(author_verses, chunk_size=300, samples_per_author=5, min_words_required=None, rng=None):
    """
    Строго НЕПЕРЕСЕКАЮЩИЕСЯ сэмплы: перемешиваем бейты автора один раз
    и нарезаем поток слов на `samples_per_author` независимых кусков по `chunk_size` слов.
    """
    if rng is None:
        rng = np.random.default_rng()

    required = min_words_required or (chunk_size * samples_per_author)
    samples, labels = [], []

    for poet, verses in sorted(author_verses.items()):
        total_w = sum(len(v.split()) for v in verses)
        if total_w < required:
            continue

        shuffled_idx = rng.permutation(len(verses))
        word_stream = []
        for idx in shuffled_idx:
            word_stream.extend(verses[idx].split())
            if len(word_stream) >= chunk_size * samples_per_author:
                break

        for s in range(samples_per_author):
            chunk = word_stream[s * chunk_size : (s + 1) * chunk_size]
            samples.append(chunk)
            labels.append(poet)

    return samples, labels


def build_feature_matrix(samples, mode="mfw", top_k=100):
    transformed_samples = []
    global_counter = Counter()

    for words in samples:
        if mode == "mfw":
            tokens = [normalize_for_mfw(w) for w in words]
        elif mode == "arud":
            tokens = [encode_word_arud_strict(w) for w in words]
        elif mode == "arud_2gram":
            # Дополнительно: биграммы Аруд-кодов слов (захватывают стыки стоп внутри полустишия)
            uni = [encode_word_arud_strict(w) for w in words]
            uni = [t for t in uni if t]
            tokens = [f"{uni[k]}_{uni[k+1]}" for k in range(len(uni) - 1)]
        else:
            raise ValueError(mode)

        tokens = [t for t in tokens if t]
        transformed_samples.append(tokens)
        global_counter.update(tokens)

    vocab = [tok for tok, _ in global_counter.most_common(top_k)]
    matrix = np.zeros((len(samples), len(vocab)), dtype=np.float64)

    for i, tokens in enumerate(transformed_samples):
        if not tokens:
            continue
        c = Counter(tokens)
        total = len(tokens)
        for j, feat in enumerate(vocab):
            matrix[i, j] = c[feat] / total

    return matrix, vocab


def burrows_delta_ward_eval(matrix, labels):
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    z_scores = (matrix - mean) / std

    condensed_dist = pdist(z_scores, metric="cityblock") / z_scores.shape[1]
    Z = linkage(condensed_dist, method="ward")

    n_clusters = len(set(labels))
    pred_clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    return rand_score(labels, pred_clusters), adjusted_rand_score(labels, pred_clusters), vocab_size(matrix)

def vocab_size(matrix):
    return matrix.shape[1]


def run_benchmark(author_verses, chunk_size, samples_per_author, n_runs=10, seed=42):
    rng = np.random.default_rng(seed)
    modes = ["arud", "arud_2gram", "mfw"]
    results = {m: {"ri": [], "ari": [], "n_vocab": 0} for m in modes}
    n_authors = 0

    for _ in range(n_runs):
        samples, labels = sample_non_overlapping_chunks(
            author_verses, chunk_size=chunk_size, samples_per_author=samples_per_author, rng=rng
        )
        n_authors = len(set(labels))
        for m in modes:
            mat, vocab = build_feature_matrix(samples, mode=m, top_k=100)
            ri, ari, n_v = burrows_delta_ward_eval(mat, labels)
            results[m]["ri"].append(ri)
            results[m]["ari"].append(ari)
            results[m]["n_vocab"] = n_v

    print(f"\n--- NON-OVERLAPPING: {n_authors} авторов | {samples_per_author} сэмплов x {chunk_size} слов ---")
    for m in modes:
        ri_m, ri_s = np.mean(results[m]["ri"]), np.std(results[m]["ri"])
        ari_m, ari_s = np.mean(results[m]["ari"]), np.std(results[m]["ari"])
        v_sz = results[m]["n_vocab"]
        print(f"  {m:10s} (Vocab={v_sz:3d}) -> Mean RI: {ri_m:.4f} (±{ri_s:.4f}) | Mean ARI: {ari_m:.4f} (±{ari_s:.4f})")


if __name__ == "__main__":
    author_verses = load_corpus("data/raw/diwans")

    print("=== ПРОВЕРКА ИСПРАВЛЕННОГО КОДИРОВАНИЯ БЕЙТА ===")
    for w in author_verses["abid-ibn-al-abras"][0].split():
        print(f"{w:15s} -> MFW: {normalize_for_mfw(w):10s} | Arud: {encode_word_arud_strict(w)}")

    # Посмотрим, сколько вообще уникальных паттернов Аруда существует после фикса '00'
    all_samples, _ = sample_non_overlapping_chunks(author_verses, chunk_size=300, samples_per_author=5)
    _, vocab_arud = build_feature_matrix(all_samples, mode="arud", top_k=500)
    print(f"\nВсего уникальных валидных Arud-паттернов слов в корпусе: {len(vocab_arud)}")
    print("Топ-15 паттернов:", vocab_arud[:15])

    # Сценарий 1: Все 10 авторов (максимальный размер без пересечений для Хариса = 5 x 300 = 1500 слов)
    run_benchmark(author_verses, chunk_size=300, samples_per_author=5, n_runs=20)
    run_benchmark(author_verses, chunk_size=150, samples_per_author=10, n_runs=20)

    # Сценарий 2: Топ-6 авторов (>= 5000 слов), строго по статье: 10 сэмплов по 500 слов и 5 сэмплов по 1000 слов
    run_benchmark(author_verses, chunk_size=500, samples_per_author=10, n_runs=20)
    run_benchmark(author_verses, chunk_size=1000, samples_per_author=5, n_runs=20)