# Уведомления о сторонних компонентах (THIRD-PARTY NOTICES)

Проект VoiceAI распространяется под лицензией MIT (см. `LICENSE`). Ниже перечислены
сторонние библиотеки и компоненты, которые используются при сборке и работе, с их
лицензиями. Полные тексты лицензий находятся в исходных кодах соответствующих
крейтов (в кеше cargo) и по ссылкам ниже.

## Rust-зависимости (Cargo.toml)

| Крейт | Назначение | Лицензия |
| --- | --- | --- |
| `whisper-rs` / `whisper-rs-sys` | Привязки Rust к whisper.cpp (распознавание речи) | Unlicense (public domain). Нижележащий `whisper.cpp` — MIT |
| `transcribe-rs` | Распознавание моделью GigaAM через ONNX Runtime | MIT |
| `ort` / `ort-sys` | Привязки к ONNX Runtime (используются `transcribe-rs`) | MIT OR Apache-2.0. Сам ONNX Runtime — MIT |
| `cpal` | Захват звука с микрофона | Apache-2.0 |
| `hound` | Чтение/запись WAV | Apache-2.0 |
| `global-hotkey` | Глобальные горячие клавиши | Apache-2.0 OR MIT |
| `eframe` / `egui` | Графический интерфейс | MIT OR Apache-2.0 |
| `serde` / `serde_json` | Сериализация настроек и журнала | MIT OR Apache-2.0 |
| `chrono` | Дата и время записей журнала | MIT OR Apache-2.0 |
| `uuid` | Идентификаторы записей журнала | MIT OR Apache-2.0 |
| `arboard` | Буфер обмена (запасной путь вставки, Windows) | MIT OR Apache-2.0 |
| `tray-icon` | Иконка в системном трее (Windows) | MIT OR Apache-2.0 |
| `winreg` | Автозапуск через реестр Windows | MIT |
| `windows-sys` | Win32 API (SendInput, MessageBox и др.) | MIT OR Apache-2.0 |

## Модели распознавания

- `ggml-large-v3-turbo.bin` — веса Whisper large-v3-turbo в формате GGML из
  репозитория [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp).
  Исходная модель — OpenAI Whisper, лицензия MIT.
- GigaAM v3 e2e-ctc (`model.onnx`, `vocab.txt`) — модель SberDevices (GigaChat
  Team) в формате ONNX из репозитория
  [istupakov/gigaam-v3-onnx](https://huggingface.co/istupakov/gigaam-v3-onnx),
  лицензия MIT. В поставку не входит: пользователь скачивает её сам из окна
  настроек.

## Системные/динамические компоненты (не входят в исходный код)

- **NVIDIA CUDA Runtime, cuBLAS, cuBLAS-Lt** — динамические библиотеки, которые
  копируются рядом с исполняемым файлом для GPU-ускорения на Windows. Распространяются
  под лицензионным соглашением NVIDIA (EULA), устанавливаются CUDA Toolkit:
  <https://docs.nvidia.com/cuda/eula/>.
- **Microsoft Visual C++ / MSVC, LLVM/Clang (libclang)** — только инструменты
  сборки, в дистрибутив приложения не входят.

## Свои компоненты

Код VoiceAI (исходные файлы `src/`, `examples/`, скрипты сборки) написан для этого
проекта и распространяется под лицензией MIT.
