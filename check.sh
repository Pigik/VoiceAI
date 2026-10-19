#!/usr/bin/env bash
# Локальная проверка VoiceAI одной командой: формат + clippy + тесты.
# Те же шаги выполняет CI (.github/workflows/ci.yml).
#
# Использование (Linux / macOS):
#     bash check.sh
#
# Завершается кодом 0, только если все три шага прошли.
set -u
cd "$(dirname "$0")"

failed=0

step() {
    echo
    echo "== $1 =="
}

step "1/3. Форматирование (cargo fmt --check)"
if ! cargo fmt --all -- --check; then failed=1; fi

step "2/3. Статический анализ (cargo clippy -D warnings)"
if ! cargo clippy --all-targets -- -D warnings; then failed=1; fi

step "3/3. Тесты (cargo test --all-targets)"
if ! cargo test --all-targets; then failed=1; fi

echo
if [ "$failed" -ne 0 ]; then
    echo "Проверка завершилась с ошибками." >&2
    exit 1
fi
echo "Готово: формат, clippy и тесты прошли."
exit 0
