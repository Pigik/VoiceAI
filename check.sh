#!/usr/bin/env bash
# Локальный чекер VoiceAI: формат + clippy + тесты одной командой (T-15, T-16).
#
# Использование (Linux / macOS):
#     ./check.sh
#
# Завершается кодом 0, только если все три шага прошли.
set -u

failed=0

step() {
    echo
    echo "== $1 =="
}

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