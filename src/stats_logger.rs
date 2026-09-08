use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReplicaLog {
    pub id: Uuid,                   // T-13: Уникальный ID
    pub timestamp: DateTime<Local>, // T-12: Время и часовой пояс
    pub app_name: String,           // T-06: Приложение-цель
    pub duration_ms: u64,           // T-05: Задержка/длительность
    pub word_count: usize,          // T-04: Счетчик слов
    pub text: Option<String>,       // T-10: Текст (None в режиме приватности)
    pub wpm: f64,                   // Скорость диктовки
}

const MAX_REPLICAS: usize = 200; // T-11

pub fn log_replica(log_path: &std::path::Path, mut entry: ReplicaLog, privacy_mode: bool) {
    if privacy_mode {
        entry.text = None;
    }

    // T-01: Сериализация в одну строку
    let Ok(json_line) = serde_json::to_string(&entry) else {
        return;
    };
    let json_line = format!("{json_line}\n");

    // Добавление в конец файла (ошибки молча пропускаем — журнал не ломает запись).
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = file.write_all(json_line.as_bytes());
    }

    // Ротация (Очистка до MAX_REPLICAS)
    enforce_log_limit(log_path);
}

/// Читает все сохранённые записи из журнала (пустой список, если файла нет).
pub fn read_replicas(path: &std::path::Path) -> Vec<ReplicaLog> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let reader = BufReader::new(file);
    reader
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str::<ReplicaLog>(&line).ok())
        .collect()
}

// Временная сложность: O(N), Пространственная сложность: O(N), где N = 200.
fn enforce_log_limit(path: &std::path::Path) {
    let Ok(file) = std::fs::File::open(path) else {
        return;
    };
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = reader.lines().map_while(Result::ok).collect();

    if lines.len() > MAX_REPLICAS {
        let skip = lines.len() - MAX_REPLICAS;
        lines.drain(0..skip);
        let _ = std::fs::write(path, lines.join("\n") + "\n");
    }
}

/// Перезаписывает файл журнала целиком списком записей (одна строка на запись).
///
/// Используется при удалении отдельных записей и при полной очистке журнала.
pub fn rewrite_replicas(path: &std::path::Path, entries: &[ReplicaLog]) {
    let Ok(mut file) = std::fs::File::create(path) else {
        return;
    };
    for entry in entries {
        if let Ok(json_line) = serde_json::to_string(entry) {
            let _ = writeln!(file, "{json_line}");
        }
    }
}

/// Полностью удаляет файл журнала (все записи с диска).
pub fn clear_replicas(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_and_read_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("voiceai_test_replicas.jsonl");
        let _ = std::fs::remove_file(&path);

        let entry = ReplicaLog {
            id: Uuid::new_v4(),
            timestamp: Local::now(),
            app_name: "Тест".to_string(),
            duration_ms: 1200,
            word_count: 5,
            text: Some("привет мир".to_string()),
            wpm: 250.0,
        };
        log_replica(&path, entry.clone(), false);
        let read = read_replicas(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].word_count, 5);
        assert_eq!(read[0].text.as_deref(), Some("привет мир"));
    }

    #[test]
    fn privacy_mode_strips_text() {
        let dir = std::env::temp_dir();
        let path = dir.join("voiceai_test_replicas_privacy.jsonl");
        let _ = std::fs::remove_file(&path);

        let entry = ReplicaLog {
            id: Uuid::new_v4(),
            timestamp: Local::now(),
            app_name: "Тест".to_string(),
            duration_ms: 1000,
            word_count: 3,
            text: Some("секретный текст".to_string()),
            wpm: 180.0,
        };
        log_replica(&path, entry, true);
        let read = read_replicas(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(read.len(), 1);
        assert!(read[0].text.is_none());
        assert_eq!(read[0].word_count, 3);
    }

    #[test]
    fn rewrite_and_clear() {
        let dir = std::env::temp_dir();
        let path = dir.join("voiceai_test_replicas_rewrite.jsonl");
        let _ = std::fs::remove_file(&path);

        let make = |words: usize, text: &str| ReplicaLog {
            id: Uuid::new_v4(),
            timestamp: Local::now(),
            app_name: "Тест".to_string(),
            duration_ms: 1000,
            word_count: words,
            text: Some(text.to_string()),
            wpm: 180.0,
        };
        let a = make(1, "один");
        let b = make(2, "два");

        log_replica(&path, a.clone(), false);
        log_replica(&path, b.clone(), false);
        assert_eq!(read_replicas(&path).len(), 2);

        // Перезапись без первой записи — как при её удалении.
        rewrite_replicas(&path, std::slice::from_ref(&b));
        let rest = read_replicas(&path);
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].text.as_deref(), Some("два"));

        // Полная очистка — файл исчезает, чтение даёт пустой список.
        clear_replicas(&path);
        assert!(!path.exists());
        assert!(read_replicas(&path).is_empty());
    }
}
