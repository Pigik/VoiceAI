//! Сводная статистика журнала реплик для окна «Журнал».

use crate::stats_logger::ReplicaLog;

/// Статистика записей за сегодняшний день.
pub struct DailyStats {
    /// Сколько записей сделано сегодня.
    pub total_replicas: usize,
    /// Сколько слов распознано сегодня.
    pub total_words: usize,
    /// Средняя длительность записи, мс.
    pub avg_duration_ms: f64,
}

/// Считает статистику по записям, сделанным сегодня (по местному времени).
pub fn calculate_daily_stats(logs: &[ReplicaLog]) -> DailyStats {
    let today = chrono::Local::now().date_naive();
    let daily_logs: Vec<&ReplicaLog> = logs
        .iter()
        .filter(|l| l.timestamp.date_naive() == today)
        .collect();

    let total_replicas = daily_logs.len();
    let total_words: usize = daily_logs.iter().map(|l| l.word_count).sum();
    let total_duration: u64 = daily_logs.iter().map(|l| l.duration_ms).sum();
    let avg_duration_ms = if total_replicas > 0 {
        total_duration as f64 / total_replicas as f64
    } else {
        0.0
    };

    DailyStats {
        total_replicas,
        total_words,
        avg_duration_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Local, TimeZone};
    use uuid::Uuid;

    fn replica(timestamp: DateTime<Local>, words: usize, duration_ms: u64) -> ReplicaLog {
        ReplicaLog {
            id: Uuid::new_v4(),
            timestamp,
            app_name: String::new(),
            duration_ms,
            word_count: words,
            text: None,
        }
    }

    /// В сводку попадают только сегодняшние записи.
    #[test]
    fn counts_only_today() {
        let old = Local
            .with_ymd_and_hms(2000, 1, 1, 12, 0, 0)
            .single()
            .expect("корректная дата");
        let logs = [
            replica(Local::now(), 3, 1000),
            replica(Local::now(), 5, 3000),
            replica(old, 100, 9000),
        ];
        let stats = calculate_daily_stats(&logs);
        assert_eq!(stats.total_replicas, 2);
        assert_eq!(stats.total_words, 8);
        assert!((stats.avg_duration_ms - 2000.0).abs() < 1e-9);
    }

    #[test]
    fn empty_log_gives_zeroes() {
        let stats = calculate_daily_stats(&[]);
        assert_eq!(stats.total_replicas, 0);
        assert_eq!(stats.avg_duration_ms, 0.0);
    }
}
