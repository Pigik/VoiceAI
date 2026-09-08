# Уведомления о сторонних компонентах (THIRD-PARTY NOTICES)

Проект VoiceAI распространяется под лицензией MIT (см. `LICENSE`). Ниже перечислены
сторонние библиотеки и компоненты, которые используются при сборке и работе, с их
лицензиями. Полные тексты лицензий находятся в исходных кодах соответствующих
крейтов (в кеше cargo) и по ссылкам ниже.

## Rust-зависимости (Cargo.toml)

| Крейт | Назначение | Лицензия |
| --- | --- | --- |
| `whisper-rs` / `whisper-rs-sys` | Привязки Rust к whisper.cpp (распознавание речи) | Unlicense (public domain). Нижележащий `whisper.cpp` — MIT |
| `cpal` | Захват звука с микрофона | Apache-2.0 |
| `hound` | Чтение/запись WAV | Apache-2.0 |
| `global-hotkey` | Глобальные горячие клавиши | Apache-2.0 OR MIT |
| `eframe` / `egui` | Графический интерфейс | MIT OR Apache-2.0 |
| `serde` / `serde_json` | Сериализация настроек | MIT OR Apache-2.0 |
| `arboard` | Системный буфер обмена (запасной путь вставки) | MIT OR Apache-2.0 |
| `tray-icon` | Иконка в системном трее (Windows) | MIT OR Apache-2.0 |
| `winreg` | Автозапуск через реестр Windows | MIT |
| `windows-sys` | Win32 API (SendInput, MessageBox и др.) | MIT OR Apache-2.0 |

## Модель Whisper

- `ggml-large-v3-turbo.bin` — веса модели из репозитория
  [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp),
  распространяются под лицензией MIT (автор первоначальной модели — OpenAI,
  лицензия MIT).

## Системные/динамические компоненты (не входят в исходный код)

- **NVIDIA CUDA Runtime, cuBLAS, cuBLAS-Lt** — динамические библиотеки, которые
  копируются рядом с исполняемым файлом для GPU-ускорения на Windows. Распространяются
  под лицензионным соглашением NVIDIA (EULA), устанавливаются CUDA Toolkit:
  <https://docs.nvidia.com/cuda/eula/>.
- **Microsoft Visual C++ Redistributable / MSVC, LLVM/Clang (libclang)** — только
  инструменты сборки, в дистрибутив приложения не входят.

## Свои компоненты

Код VoiceAI (исходные файлы `src/`, `examples/`, скрипты сборки) написан для этого
проекта с нуля и распространяется под лицензией MIT.