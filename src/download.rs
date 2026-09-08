//! Скачивание файлов моделей через curl c прогрессом.
//!
//! Windows 10+ и macOS поставляют curl в системе; в таком виде не нужно тащить
//! в проект сетевой стек (TLS и т.п.) ради редкого скачивания модели. Прогресс
//! передаётся в вызывающий код через замыкание, чтобы GUI мог показывать полосу.

use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

/// Интервал опроса размера скачиваемого файла (для обновления прогресса).
const POLL_INTERVAL: Duration = Duration::from_millis(120);

/// Доступен ли `curl` в системе (Windows 10+ и macOS — да, по умолчанию).
fn curl_available() -> bool {
    Command::new("curl")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Размер удалённого файла в байтах через HTTP HEAD (0 — размер неизвестен).
pub fn remote_size(url: &str) -> u64 {
    let Ok(output) = Command::new("curl").args(["-sIL", url]).output() else {
        return 0;
    };
    let text = String::from_utf8_lossy(&output.stdout);
    parse_content_length(&text)
}

/// Вытаскивает последний заголовок `content-length` из вывода curl -I
/// (по всей цепочке редиректов берём финальный размер).
fn parse_content_length(text: &str) -> u64 {
    text.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            if key.trim().eq_ignore_ascii_case("content-length") {
                value.trim().parse().ok()
            } else {
                None
            }
        })
        .next_back()
        .unwrap_or(0)
}

/// Скачивает файл по URL и сохраняет его в `dest` (блокирующая функция).
///
/// Сначала идёт во временный файл `<dest>.part`, чтобы оборванное
/// скачивание не оставило «наполовину готовую» модель; по завершении файл
/// переименовывается в `dest`. `on_progress` вызывается с парами
/// (скачано байт, всего байт; всего = 0, если размер неизвестен).
pub fn download_file(url: &str, dest: &Path, on_progress: impl Fn(u64, u64)) -> Result<(), String> {
    if !curl_available() {
        return Err(
            "Для скачивания моделей нужен curl (в Windows 10+ и macOS он уже установлен)"
                .to_string(),
        );
    }

    let parent = dest
        .parent()
        .ok_or_else(|| format!("Некорректный путь для скачивания: {}", dest.display()))?;
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("Не удалось создать папку {}: {err}", parent.display()))?;

    let total = remote_size(url);

    let mut tmp_name = dest
        .file_name()
        .ok_or_else(|| format!("Некорректное имя файла: {}", dest.display()))?
        .to_os_string();
    tmp_name.push(".part");
    let tmp = dest.with_file_name(tmp_name);

    // -sS: без прогресс-бара самого curl, но с выводом ошибок;
    // --fail: ненулевой код при 4xx/5xx; --location: идём за редиректами
    // (huggingface отдаёт файлы через редирект на CDN).
    let mut child = Command::new("curl")
        .args(["--fail", "--location", "--silent", "--show-error"])
        .arg("--output")
        .arg(&tmp)
        .arg(url)
        .spawn()
        .map_err(|err| format!("Не удалось запустить curl: {err}"))?;

    while !finished(&mut child) {
        let downloaded = std::fs::metadata(&tmp).map(|meta| meta.len()).unwrap_or(0);
        on_progress(downloaded, total);
        thread::sleep(POLL_INTERVAL);
    }

    let status = child
        .wait()
        .map_err(|err| format!("Ошибка ожидания curl: {err}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("Скачивание не удалось (curl: {status})"));
    }

    let len = std::fs::metadata(&tmp).map(|meta| meta.len()).unwrap_or(0);
    if len == 0 {
        let _ = std::fs::remove_file(&tmp);
        return Err("Скачанный файл оказался пустым".to_string());
    }
    if total > 0 && len < total {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!(
            "Файл скачан не полностью: {} из {} байт",
            len, total
        ));
    }

    std::fs::rename(&tmp, dest)
        .map_err(|err| format!("Не удалось сохранить файл {}: {err}", dest.display()))?;
    on_progress(len, total);
    Ok(())
}

/// Закончил ли curl работать (true — дочерний процесс вышел).
fn finished(child: &mut std::process::Child) -> bool {
    match child.try_wait() {
        Ok(Some(_)) => true,
        Ok(None) => false,
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Размер вытаскивается из последнего заголовка даже при редиректах.
    #[test]
    fn content_length_parsed() {
        let headers = "HTTP/1.1 302 Found\r\ncontent-length: 10\r\n\r\n\
                       HTTP/1.1 200 OK\r\nContent-Length: 12345\r\n\r\n";
        assert_eq!(parse_content_length(headers), 12345);
    }

    #[test]
    fn content_length_missing_means_zero() {
        assert_eq!(parse_content_length("HTTP/1.1 200 OK\r\n\r\n"), 0);
        assert_eq!(parse_content_length(""), 0);
    }
}
