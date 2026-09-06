use crate::stats_logger::ReplicaLog;
use std::collections::HashMap;

pub struct DailyStats {
    pub total_replicas: usize,                     // T-03
    pub total_words: usize,                        // T-04
    pub avg_latency_ms: f64,                       // T-05
    pub apps_distribution: HashMap<String, usize>, // T-06
}

pub fn calculate_daily_stats(logs: &[ReplicaLog]) -> DailyStats {
    let today = chrono::Local::now().date_naive();
    let daily_logs: Vec<_> = logs
        .iter()
        .filter(|l| l.timestamp.date_naive() == today)
        .collect();

    let total_replicas = daily_logs.len();
    let total_words: usize = daily_logs.iter().map(|l| l.word_count).sum();

    let total_latency: u64 = daily_logs.iter().map(|l| l.duration_ms).sum();
    let avg_latency_ms = if total_replicas > 0 {
        total_latency as f64 / total_replicas as f64
    } else {
        0.0
    };

    let mut apps_distribution = HashMap::new();
    for log in &daily_logs {
        *apps_distribution.entry(log.app_name.clone()).or_insert(0) += 1;
    }

    DailyStats {
        total_replicas,
        total_words,
        avg_latency_ms,
        apps_distribution,
    }
}

// T-08: Экспорт в CSV
pub fn export_to_csv(logs: &[ReplicaLog], out_path: &str) -> Result<(), csv::Error> {
    let mut wtr = csv::Writer::from_path(out_path)?;
    for log in logs {
        wtr.serialize(log)?;
    }
    wtr.flush()?;
    Ok(())
}

// T-09: Сброс статистики (удаление файла)
pub fn reset_statistics(log_path: &std::path::Path) {
    let _ = std::fs::remove_file(log_path);
}
