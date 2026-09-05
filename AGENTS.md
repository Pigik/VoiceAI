# AGENTS.md

Документация для LLM-агентов (opencode и т.п.), работающих над этим репозиторием.
Читай перед началом любых изменений.

## Что это за проект

**VoiceAI** — настольный «диктофон с распознаванием речи» на Rust (Windows / macOS / Linux).
Пользователь зажимает глобальную горячую клавишу **F1**, говорит, отпускает — программа
локально (без интернета) распознаёт речь через Whisper (whisper.cpp) и вставляет
текст в активное приложение под курсором.

Ключевые свойства:

- **Локальность**: модель Whisper и вся обработка — на машине пользователя, CUDA (Windows) / Metal (macOS) / CPU (Linux).
- **Надёжность**: фоновые потоки никогда не «умирают» (catch_unwind), ошибки пишутся в `voiceai.log`, сбой одной записи не ломает следующие.
- **Кросс-платформенность**: есть Windows-специфичный код (вставка текста, трей, автозапуск, хоткей), отключаемый через `#[cfg(target_os = "windows")]`.

## Как проверить работу (команды)

Быстрая итерация — **`cargo check`** (не линкует, сильно быстрее `cargo build`):

```powershell
# PowerShell, из корня проекта:
cargo check
```

Полезно знать: `cargo` может быть недоступен по имени в вашем shell — он лежит в
`%USERPROFILE%\.cargo\bin\cargo.exe`. Полный релиз с CUDA-окружением — `.\build.bat`.

Полный набор проверок (та же логика в CI):

```powershell
cargo fmt --all -- --check    # форматирование
cargo clippy --all-targets -- -D warnings   # статический анализ (без предупреждений)
cargo test                    # все тесты
```

Модульные тесты — `cargo test --lib`; одиночный модуль — `cargo test --lib insert`
и т.п. Тесты для каждой платформы идут в CI (Windows/Ubuntu 24.04/macOS 15, см.
`.github/workflows/ci.yml`).

Пример без GUI (полный конвейер: модель → аудио → распознавание → пост-обработка):

```powershell
cargo run --example transcribe_test -- test_russian.wav rу
```

## Архитектура и модули

Вся нетривиальная логика живёт в библиотеке `voiceai` (`src/lib.rs`),
бинарник — `src/main.rs`.

| Файл | Что это | Ключевое |
|---|---|---|
| `src/main.rs` | GUI (eframe/egui), цикл записи, транскрибация, вставка | Точка входа; `run_recorder`, `run_transcriber`, `transcribe_and_save`, `resolve_model_path` |
| `src/insert.rs` | Вставка текста в активное окно (Windows) | **SendInput + KEYEVENTF_UNICODE, буфер обмена НЕ используется** |
| `src/audio.rs` | Подготовка аудио | `resample_to_whisper`, `trim_silence`, `apply_noise_reduction`, `normalize_audio`, `has_speech_energy` |
| `src/postprocess.rs` | Очистка текста | слова-паразиты, пунктуация, абзацы (`postprocess_text`) |
| `src/settings.rs` | Настройки + `config.json` | `Settings`, `PressMode`, `LANGUAGE_CHOICES`, `load_settings`/`save_settings` |
| `src/autostart.rs` | Автозапуск в Windows (реестр) | — |
| `src/tray.rs` | Иконка в трее (Windows) | `create_tray`, `update_tray` |
| `src/single_instance.rs` | Одна копия приложения (порт-лок) | `SingleInstanceGuard` |
| `build.rs` | Копирует модель + CUDA DLL рядом с exe (Windows) | — |
| `examples/transcribe_test.rs` | Проверка распознавания без GUI | — |

## Поведение приложения, которое нельзя ломать

- **Запись**: F1 старт/стоп (режимы `PushToTalk`/`Toggle`). После отпускания дослушивается
  «хвост» 200 мс, запись сохраняется в `output.wav`, уходит в фоновый поток транскрибации.
- **Вставка текста — только на Windows и без буфера обмена**: `insert::insert_text_at_cursor`
  печатает текст через `SendInput` с `KEYEVENTF_UNICODE`. **НЕ возвращай это на clipboard+Ctrl+V**:
  сохранение/восстановление буфера вызывало гонку и вставку старого содержимого вместо распознанного текста.
- **Тихие записи ничего не вставляют**: перед распознаванием проверяется
  `audio::has_speech_energy`. Если речи нет (быстрое нажатие F1), возвращается `word_count = 0`,
  и вставка не происходит. Этот guard борется с «галлюцинациями» Whisper («Продолжение следует»).
- **Настройки открываются сами** при первом запуске или битом конфиге; окно настроек скроллится
  (`ScrollArea`), все элементы должны оставаться доступными без выхода за рамки окна.
- **Модель одна**: `ggml-large-v3-turbo.bin`. Запасной (квантованной) модели больше нет — не добавляй её обратно.
  Поиск: путь в настройках → `WHISPER_MODEL` → папка с exe → текущая папка → `models/`.

## Конвенции кода

- Комментарии и UI-строки — **на русском**; имена функций/переменных — по-английски.
- Без лишних комментариев в коде: комментарий объясняет *зачем*, а не *что*.
- Windows-специфичные модули и код оборачивай в `#[cfg(target_os = "windows")]`.
- Ошибки: `Result<_, Box<dyn std::error::Error>>`; в фоновых потоках — `catch_unwind`;
  отравленные мьютексы — `unwrap_or_else(|e| e.into_inner())`.
- Тесты — рядом с кодом в `#[cfg(test)] mod tests`, по одному блоку на модуль.
- Не добавляй новые зависимости без крайней необходимости; тяжёлая сборка — плата за `whisper-rs`/`eframe`.

## Известные грабли

- **`cargo build` долгий**: `whisper-rs` (feature `cuda`) компилирует весь whisper.cpp + CUDA. Для итераций используй `cargo check`.
- **`.cargo/config.toml`** хардкода пути CUDA `v13.3` (`CUDA_PATH`, `CUDA_PATH_V13_3`) — при обновлении CUDA их надо править.
- **`build.rs`** копирует `ggml-large-v3-turbo.bin` и CUDA DLL в `target/<profile>` — модель должна лежать в корне проекта.
- **Тест `single_instance::tests::second_guard_rejected_until_first_released` падает, если приложение уже запущено** (занятый порт). Это не связано с изменениями в коде.
- Сборка/линковка с `--features cuda` требует окружение MSVC (см. `build.bat`).