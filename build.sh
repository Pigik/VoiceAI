#!/usr/bin/env bash
# Сборка VoiceAI на macOS и Linux (проверено на Ubuntu 24.04 и macOS 15).
#
# Требования:
#   - Rust (rustup) — версия зафиксирована в rust-toolchain.toml
#   - C/C++ компилятор (на macOS — Xcode Command Line Tools, входит в Xcode)
#   - CMake
#   - Clang/LLVM (нужна библиотека libclang для bindgen):
#       Ubuntu: sudo apt-get install -y clang libclang-dev cmake
#   - macOS: распознавание идёт через Metal (GPU любого Mac).
#     Ubuntu: распознавание идёт на процессоре (GPU-ускорение на Linux не требуется).
set -euo pipefail
cd "$(dirname "$0")"

cargo build --release

OUT="target/release"
# Кладём модель рядом с бинарником — main.rs ищет модель в папке с программой.
if [ -f "ggml-large-v3-turbo.bin" ]; then
    cp "ggml-large-v3-turbo.bin" "$OUT/"
fi
if [ -f "ggml-large-v3-turbo-q5_0.bin" ]; then
    cp "ggml-large-v3-turbo-q5_0.bin" "$OUT/"
fi

echo
echo "Готово. Запускайте $OUT/VoiceAI"