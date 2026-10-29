#!/usr/bin/env bash
# Прогон муаллак по эпохам без присмотра: при зависании сторож завершает процесс (код 3),
# цикл перезапускает его с --resume — готовые seed берутся с диска, не обучаются заново.
#
#   nohup bash run_era_unattended.sh > logs/era_unattended.out 2>&1 &
#   tail -f logs/era_unattended.out          # смотреть прогресс; закрыть терминал можно
#
# Параметры через переменные окружения: SEEDS (3), SCHEME (coarse), WATCHDOG_MIN (20), TRIES (6).
set -u
cd "$(dirname "$0")"
mkdir -p logs
export HF_HUB_OFFLINE=1
SEEDS=${SEEDS:-3}
SCHEME=${SCHEME:-coarse}
WATCHDOG_MIN=${WATCHDOG_MIN:-20}
TRIES=${TRIES:-6}

for try in $(seq 1 "$TRIES"); do
  echo "=== $(date '+%H:%M:%S') попытка $try/$TRIES: run_era_cases --seeds $SEEDS --scheme $SCHEME --resume"
  python -m aa.era.run_era_cases --scheme "$SCHEME" --seeds "$SEEDS" --resume --watchdog-min "$WATCHDOG_MIN"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "=== $(date '+%H:%M:%S') готово; собираю отчёт"
    python -m aa.era.report --dir "data/processed/era_$SCHEME"
    exit 0
  fi
  echo "=== $(date '+%H:%M:%S') прогон завершился с кодом $code (3 = сработал сторож) — перезапуск через 30 с"
  sleep 30
done
echo "=== не удалось за $TRIES попыток — см. logs/era_cases_${SCHEME}_*.log"
exit 1
