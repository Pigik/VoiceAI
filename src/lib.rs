//! Библиотечная часть проекта VoiceAI.
//!
//! Здесь живёт логика, не привязанная к окну программы: подготовка аудио,
//! пост-обработка текста, настройки, журнал, модели и т.п. Её используют
//! основное приложение (`src/main.rs`), пример `examples/transcribe_test.rs`
//! и модульные тесты.

pub mod analytics;
pub mod audio;
pub mod autostart;
pub mod download;
pub mod models;
pub mod postprocess;
pub mod settings;
pub mod single_instance;
pub mod stats_logger;
pub mod whisper_config;

#[cfg(target_os = "windows")]
pub mod insert;

#[cfg(target_os = "windows")]
pub mod tray;
