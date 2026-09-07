//! Пользовательские настройки приложения и их хранение на диске.
//!
//! Настройки сериализуются в человекочитаемый JSON (см. `config_path`),
//! поэтому файл можно править текстовым редактором, а схема переживает
//! обновление программы: новые поля получают значения по умолчанию.
//! Хранится файл вне папки с программой (см. раздел «Настройки» в README).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Режим нажатия клавиши записи.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressMode {
    /// «Рация»: запись идёт, пока клавиша зажата.
    #[default]
    PushToTalk,
    /// «Включил/выключил»: каждое нажатие клавиши переключает запись.
    Toggle,
}

/// Доступные языки распознавания: код ISO и подпись для интерфейса.
/// «auto» — автоопределение языка Whisper.
pub const LANGUAGE_CHOICES: &[(&str, &str)] = &[
    ("auto", "Автоопределение"),
    ("ru", "Русский"),
    ("en", "Английский"),
    ("uk", "Украинский"),
    ("de", "Немецкий"),
    ("fr", "Французский"),
    ("es", "Испанский"),
    ("zh", "Китайский"),
];

/// Возвращает подпись языка для интерфейса; для незнакомого кода — сам код.
pub fn language_label(lang: &str) -> String {
    LANGUAGE_CHOICES
        .iter()
        .find(|(code, _)| *code == lang)
        .map(|(_, label)| label.to_string())
        .unwrap_or_else(|| lang.to_string())
}

fn default_language() -> String {
    "ru".to_string()
}

fn default_auto_punctuation() -> bool {
    true
}

fn default_input_gain() -> f32 {
    1.0
}

fn default_keep_audio() -> bool {
    true
}

/// Пользовательские настройки приложения. Сериализуются в JSON в
/// `config.json` и переживают обновление программы.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    /// Выбранное устройство ввода (по имени). `None` — по умолчанию.
    #[serde(default)]
    pub input_device: Option<String>,
    /// Режим нажатия клавиши записи.
    #[serde(default)]
    pub press_mode: PressMode,
    /// Автоматическая пунктуация распознанного текста (точки, абзацы).
    #[serde(default = "default_auto_punctuation")]
    pub auto_punctuation: bool,
    /// Язык распознавания: код ISO или «auto» для автоопределения.
    #[serde(default = "default_language")]
    pub language: String,
    /// Путь к файлу модели Whisper. `None` — автоматический поиск
    /// (переменная WHISPER_MODEL, папка с программой, текущая папка, models/).
    #[serde(default)]
    pub model_path: Option<String>,
    /// Включать ли приложение при входе в систему.
    #[serde(default)]
    pub auto_start: bool,
    /// Усиление входного сигнала (1.0 — без изменений).
    #[serde(default = "default_input_gain")]
    pub input_gain: f32,
    /// Подавлять ли фоновый шум при записи.
    #[serde(default)]
    pub noise_reduction: bool,
    /// Сохранять ли аудиофайл записи (output.wav) после распознавания.
    /// `false` — запись расшифровывается «из памяти», файл не создаётся.
    #[serde(default = "default_keep_audio")]
    pub keep_audio: bool,

    #[serde(default)]
    pub privacy_mode: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            input_device: None,
            press_mode: PressMode::PushToTalk,
            auto_punctuation: true,
            language: default_language(),
            model_path: None,
            auto_start: false,
            input_gain: default_input_gain(),
            noise_reduction: false,
            keep_audio: default_keep_audio(),
            privacy_mode: false,
        }
    }
}

/// Результат чтения настроек с диска с указанием «обстоятельств».
pub struct LoadOutcome {
    /// Настройки: из файла, или значения по умолчанию при ошибке/первом запуске.
    pub settings: Settings,
    /// Понятное описание проблемы, если файл есть, но не читается
    /// или содержит некорректные значения.
    pub error: Option<String>,
    /// Файла настроек ещё не было — это первый запуск программы.
    pub first_launch: bool,
}

/// Папка с данными пользователя для этого приложения.
///
/// Windows: `%APPDATA%\VoiceAI`, macOS/Linux: `~/.config/VoiceAI`. Хранится
/// вне папки с программой, поэтому обновление приложения поверх старой версии
/// не сбрасывает настройки (см. раздел «Установка поверх...» в README).
pub fn config_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        let appdata = std::env::var("APPDATA").ok()?;
        Some(PathBuf::from(appdata).join("VoiceAI"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let base = std::env::var("XDG_CONFIG_HOME")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var("HOME").ok().map(|home| {
                    let mut home = PathBuf::from(home);
                    home.push(".config");
                    home
                })
            })?;
        Some(base.join("VoiceAI"))
    }
}

/// Полный путь к файлу настроек `config.json`.
pub fn config_path() -> Option<PathBuf> {
    config_dir().map(|dir| dir.join("config.json"))
}

/// Загружает настройки с диска. Если файла нет — фиксируется первый запуск;
/// если файл испорчен — понятная ошибка и значения по умолчанию.
pub fn load_settings() -> LoadOutcome {
    let Some(path) = config_path() else {
        return LoadOutcome {
            settings: Settings::default(),
            error: None,
            first_launch: true,
        };
    };

    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return LoadOutcome {
                settings: Settings::default(),
                error: None,
                first_launch: true,
            };
        }
        Err(err) => {
            return LoadOutcome {
                settings: Settings::default(),
                error: Some(format!(
                    "Не удалось прочитать файл настроек\n{}\n\n{err}",
                    path.display()
                )),
                first_launch: false,
            };
        }
    };

    match parse_settings(&text) {
        Ok(settings) => LoadOutcome {
            settings,
            error: None,
            first_launch: false,
        },
        Err(message) => LoadOutcome {
            settings: Settings::default(),
            error: Some(format!(
                "Ошибка в файле настроек\n{}\n\n{message}\n\nИспользуются значения по умолчанию. \
                 Исправьте файл или нажмите «Сбросить настройки».",
                path.display()
            )),
            first_launch: false,
        },
    }
}

/// Разбирает JSON-текст настроек. Возвращает понятное сообщение об ошибке,
/// если файл синтаксически повреждён или содержит некорректные значения.
pub fn parse_settings(text: &str) -> Result<Settings, String> {
    serde_json::from_str::<Settings>(text).map_err(describe_config_error)
}

/// Превращает ошибку serde_json в человекочитаемый текст.
fn describe_config_error(err: serde_json::Error) -> String {
    use serde_json::error::Category;
    let location = format!("строка {} столбец {}", err.line(), err.column());
    match err.classify() {
        Category::Io => "Файл не удалось прочитать".to_string(),
        Category::Syntax => format!("Синтаксическая ошибка JSON ({location})"),
        Category::Eof => format!("Файл обрезан — не хватает конца JSON ({location})"),
        Category::Data => format!("Некорректное значение поля ({location}): {err}"),
    }
}

/// Сохраняет настройки на диск (создаёт папку при необходимости).
pub fn save_settings(settings: &Settings) -> Result<(), String> {
    let Some(path) = config_path() else {
        return Ok(());
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|err| err.to_string())?;
    }
    let json = to_json(settings)?;
    std::fs::write(&path, json)
        .map_err(|err| format!("Не удалось записать {}: {err}", path.display()))
}

/// Сериализует настройки в человекочитаемый JSON (с отступами).
pub fn to_json(settings: &Settings) -> Result<String, String> {
    serde_json::to_string_pretty(settings).map_err(|err| err.to_string())
}

/// Экспортирует профиль настроек в указанный файл.
pub fn export_settings(to_path: &str, settings: &Settings) -> Result<(), String> {
    let json = to_json(settings)?;
    std::fs::write(to_path, json).map_err(|err| format!("Не удалось записать {to_path}: {err}"))
}

/// Импортирует профиль настроек из указанного файла.
pub fn import_settings(from_path: &str) -> Result<Settings, String> {
    let text = std::fs::read_to_string(from_path)
        .map_err(|err| format!("Не удалось прочитать {from_path}: {err}"))?;
    parse_settings(&text).map_err(|message| format!("Некорректный профиль {from_path}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_config_reads_with_defaults() {
        let old = r#"{
            "input_device": "Микрофон",
            "press_mode": "Toggle",
            "auto_punctuation": false
        }"#;
        let settings = parse_settings(old).expect("старый конфиг читается");
        assert_eq!(settings.input_device.as_deref(), Some("Микрофон"));
        assert_eq!(settings.press_mode, PressMode::Toggle);
        assert!(!settings.auto_punctuation);
        assert_eq!(settings.language, "ru");
        assert_eq!(settings.model_path, None);
    }

    #[test]
    fn round_trip_keeps_values() {
        let settings = Settings {
            input_device: Some("Переговорка".to_string()),
            press_mode: PressMode::Toggle,
            auto_punctuation: false,
            language: "en".to_string(),
            model_path: Some("D:\\models\\whisper.bin".to_string()),
            auto_start: true,
            input_gain: 1.5,
            noise_reduction: true,
            keep_audio: false,
            privacy_mode: true,
        };
        let json = to_json(&settings).expect("сериализация");
        let parsed = parse_settings(&json).expect("обратная сериализация");
        assert_eq!(parsed, settings);
    }
}
