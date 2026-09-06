//! Библиотечная часть проекта VoiceAI.
//!
//! Вся нетривиальная логика (подготовка аудио и пост-обработка текста)
//! живёт здесь, чтобы её можно было переиспользовать из основного приложения
//! (`src/main.rs`) и из примеров/тестов.

pub mod analytics;
pub mod audio;
pub mod autostart;
pub mod postprocess;
pub mod settings;
pub mod single_instance;
pub mod stats_logger;

#[cfg(target_os = "windows")]
pub mod insert;

#[cfg(target_os = "windows")]
pub mod tray;
