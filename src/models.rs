//! Вторая модель распознавания — GigaAM v3 e2e-ctc (SberDevices).
//!
//! В отличие от Whisper (один файл GGML .bin), GigaAM — это папка с ONNX-моделью
//! и словарём: `model.onnx` + `vocab.txt`. Здесь описаны имена файлов, адреса
//! скачивания и правила поиска папки на диске — этот код общий для приложения
//! (`src/main.rs`) и проверочного примера (`examples/transcribe_test.rs`).
//!
//! Модель не входит в базовую поставку: пользователь скачивает её сам из окна
//! настроек (см. `download.rs`), поэтому ничего здесь не скачивается при сборке.

use std::path::PathBuf;

/// Имя папки, куда кладётся модель GigaAM.
pub const GIGAAM_DIR_NAME: &str = "giga-am-v3";
/// Имя файла ONNX-модели внутри папки (ожидается transcribe-rs).
pub const GIGAAM_MODEL_FILE: &str = "model.onnx";
/// Имя файла словаря внутри папки (ожидается transcribe-rs).
pub const GIGAAM_VOCAB_FILE: &str = "vocab.txt";

/// Адрес модели GigaAM v3 e2e-ctc в int8 (лучший выбор для CPU, ~225 МБ).
pub const GIGAAM_MODEL_URL: &str =
    "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_ctc.int8.onnx";
/// Адрес словаря модели.
pub const GIGAAM_VOCAB_URL: &str =
    "https://huggingface.co/istupakov/gigaam-v3-onnx/resolve/main/v3_e2e_ctc_vocab.txt";

/// Папки, в которых ищется модель GigaAM, по убыванию приоритета:
/// папка данных приложения (туда модель скачивается), папка с exe,
/// текущая папка, models/giga-am-v3.
pub fn gigaam_dir_candidates() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Some(config) = crate::settings::config_dir() {
        candidates.push(config.join(GIGAAM_DIR_NAME));
    }
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        candidates.push(exe_dir.join(GIGAAM_DIR_NAME));
    }
    candidates.push(PathBuf::from(GIGAAM_DIR_NAME));
    candidates.push(PathBuf::from("models").join(GIGAAM_DIR_NAME));

    candidates
}

/// Возвращает папку модели GigaAM, если в ней лежат и сама модель, и словарь.
pub fn resolve_gigaam_dir() -> Option<PathBuf> {
    gigaam_dir_candidates()
        .into_iter()
        .find(|dir| dir.join(GIGAAM_MODEL_FILE).is_file() && dir.join(GIGAAM_VOCAB_FILE).is_file())
}

/// Папка по умолчанию, куда скачивается модель GigaAM (создаётся при
/// необходимости). Приоритет — папка данных приложения (переживает обновления),
/// запасной вариант — models/giga-am-v3 в текущей папке.
pub fn gigaam_download_dir() -> Option<PathBuf> {
    if let Some(config) = crate::settings::config_dir() {
        return Some(config.join(GIGAAM_DIR_NAME));
    }
    Some(PathBuf::from("models").join(GIGAAM_DIR_NAME))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Все кандидаты на поиск — папки с именем GigaAM, начинающиеся с неё.
    #[test]
    fn all_candidates_are_gigaam_dirs() {
        let candidates = gigaam_dir_candidates();
        assert!(!candidates.is_empty());
        for dir in &candidates {
            let name = dir
                .file_name()
                .and_then(|name| name.to_str())
                .expect("имя папки существует");
            assert_eq!(name, GIGAAM_DIR_NAME);
        }
    }
}
