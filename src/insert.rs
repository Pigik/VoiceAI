//! Вставка распознанного текста в приложение, где стоит курсор.
//!
//! Текст вводится напрямую через `SendInput` с флагом `KEYEVENTF_UNICODE`:
//! каждый символ уходит в систему ввода как обычное нажатие клавиши. Такой
//! способ работает с любым приложением: браузером, Word, блокнотом и т.п.
//!
//! Буфер обмена при этом **не используется вообще**:
//! - нет гонки «восстановленный старый буфер перекрывает текст» — вставляется
//!   именно то, что нужно;
//! - буфер пользователя остаётся нетронутым (L-03) — нечего сохранять и
//!   восстанавливать, не нужна задержка на ожидание чтения буфера приложением;
//! - пустой текст не «печатается» — ничего не вставляется (L-04).
//!
//! Многострочный текст вводится как есть: переводы строк превращаются
//! в нажатие Enter (L-07).

#![cfg(target_os = "windows")]

use std::sync::Mutex;

use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, SendInput,
    VK_RETURN,
};

/// Потокобезопасность нужна, чтобы разные потоки не путали порядок
/// нажатий при одновременных вставках.
static INPUT_LOCK: Mutex<()> = Mutex::new(());

/// Событие клавиатуры: либо виртуальная клавиша (`vk`), либо Unicode-символ.
fn key_event(vk: u16, scan: u16, flags: u32) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: vk,
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

/// Печатает `text` в приложение под курсором, не трогая буфер обмена.
///
/// Возвращает `Ok(true)` при успехе или сообщение об ошибке.
pub fn insert_text_at_cursor(text: &str) -> Result<bool, Box<dyn std::error::Error>> {
    // Пустой текст печатать нечего — сразу успех (L-04).
    if text.is_empty() {
        return Ok(true);
    }

    let _guard = INPUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Для каждого символа — пара событий «нажали / отпустили».
    // Переходы строк вставляем настоящим Enter, остальное — как Unicode-символ.
    let mut inputs: Vec<INPUT> = Vec::with_capacity(text.chars().count() * 2);
    for ch in text.chars() {
        match ch {
            '\r' | '\n' => {
                inputs.push(key_event(VK_RETURN, 0, 0));
                inputs.push(key_event(VK_RETURN, 0, KEYEVENTF_KEYUP));
            }
            c => {
                inputs.push(key_event(0, c as u16, KEYEVENTF_UNICODE));
                inputs.push(key_event(0, c as u16, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
            }
        }
    }

    let sent = unsafe {
        SendInput(
            inputs.len() as u32,
            inputs.as_ptr(),
            std::mem::size_of::<INPUT>() as i32,
        )
    };

    if sent != inputs.len() as u32 {
        // Что-то не прошло (например, события отвергнуты защитой от ввода).
        // Сообщаем об ошибке — поток транскрибации запишет её в журнал.
        return Err(std::io::Error::last_os_error().into());
    }

    Ok(true)
}

/// Кладёт `text` в системный буфер обмена.
///
/// Это запасной путь (O-15): если вставка через `SendInput` не удалась
/// (окно заблокировано, защита от ввода), текст не пропадает — он попадает
/// в буфер обмена, и пользователь вставляет его вручную (Ctrl+V). Сам
/// «обычный» путь вставки буфер обмена не использует и не трогает.
pub fn copy_to_clipboard(text: &str) -> Result<(), String> {
    let mut clipboard = arboard::Clipboard::new()
        .map_err(|err| format!("Не удалось открыть буфер обмена: {err}"))?;
    clipboard
        .set_text(text.to_string())
        .map_err(|err| format!("Не удалось записать в буфер обмена: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_is_noop() {
        // L-04: пустой текст ничего не вставляет.
        assert!(insert_text_at_cursor("").is_ok());
    }

    #[test]
    fn newline_produces_return_key() {
        // Перевод строки превращается в нажатие Enter (L-07).
        let evt = key_event(VK_RETURN, 0, 0);
        unsafe {
            assert_eq!(evt.Anonymous.ki.wVk, VK_RETURN);
            assert_eq!(evt.Anonymous.ki.dwFlags, 0);
        }
    }
}
