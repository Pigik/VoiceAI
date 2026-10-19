#!/usr/bin/env bash
# Сборка VoiceAI на macOS и Linux (проверено на Ubuntu 24.04 и macOS 15).
#
# Требования:
#   - Rust (rustup) — версия зафиксирована в rust-toolchain.toml
#   - C/C++ компилятор (на macOS — Xcode Command Line Tools)
#   - CMake
#   - Clang/LLVM (библиотека libclang для bindgen):
#       Ubuntu: sudo apt-get install -y build-essential cmake clang libclang-dev libasound2-dev
#
# macOS: распознавание идёт через Metal (GPU любого Mac).
# Linux: распознавание идёт на процессоре.
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release

OUT="target/release"
# Кладём модель рядом с бинарником — программа ищет её в своей папке.
if [ -f "ggml-large-v3-turbo.bin" ]; then
    cp "ggml-large-v3-turbo.bin" "$OUT/"
fi

echo
echo "Готово. Запускайте $OUT/VoiceAI"
