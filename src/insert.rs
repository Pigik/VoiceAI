//! Вставка распознанного текста в приложение, где стоит курсор.
//!
//! Текст вводится напрямую через `SendInput` с флагом `KEYEVENTF_UNICODE`:
//! каждый символ уходит в систему ввода как обычное нажатие клавиши. Такой
//! способ работает с любым приложением: браузером, Word, блокнотом и т.п.
//!
//! Буфер обмена при этом **не используется вообще**:
//! - нет гонки «восстановленный старый буфер перекрывает текст» — вставляется
//!   именно то, что нужно;
//! - буфер пользователя остаётся нетронутым — нечего сохранять и
//!   восстанавливать, не нужна задержка на ожидание чтения буфера приложением;
//! - пустой текст не «печатается» — ничего не вставляется.
//!
//! Многострочный текст вводится как есть: переводы строк превращаются
//! в нажатие Enter.

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

/// Превращает текст в последовательность событий клавиатуры.
///
/// Для каждого символа — пара событий «нажали / отпустили». Переводы строк
/// (`\n`, `\r\n`, `\r`) вставляются одним нажатием Enter, остальные символы —
/// как Unicode-события. Символы вне базовой плоскости Unicode (эмодзи и т.п.)
/// занимают в UTF-16 два кода (суррогатную пару) — отправляем оба.
fn text_to_inputs(text: &str) -> Vec<INPUT> {
    let mut inputs: Vec<INPUT> = Vec::with_capacity(text.len() * 2);
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\r' | '\n' => {
                // «\r\n» — один перевод строки, а не два.
                if ch == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                inputs.push(key_event(VK_RETURN, 0, 0));
                inputs.push(key_event(VK_RETURN, 0, KEYEVENTF_KEYUP));
            }
            c => {
                let mut units = [0u16; 2];
                for &unit in c.encode_utf16(&mut units).iter() {
                    inputs.push(key_event(0, unit, KEYEVENTF_UNICODE));
                    inputs.push(key_event(0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
                }
            }
        }
    }
    inputs
}

/// Печатает `text` в приложение под курсором, не трогая буфер обмена.
///
/// Пустой текст ничего не вставляет и считается успехом.
pub fn insert_text_at_cursor(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }

    let _guard = INPUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let inputs = text_to_inputs(text);
    let sent = unsafe {
        SendInput(
            inputs.len() as u32,
            inputs.as_ptr(),
            std::mem::size_of::<INPUT>() as i32,
        )
    };

    if sent != inputs.len() as u32 {
        // Что-то не прошло (например, события отвергнуты защитой от ввода
        // или активно окно с повышенными правами). Ошибку запишет в журнал
        // поток транскрибации.
        return Err(std::io::Error::last_os_error().into());
    }

    Ok(())
}

/// Кладёт `text` в системный буфер обмена.
///
/// Это запасной путь: если вставка через `SendInput` не удалась
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

    /// Достаёт из событий (vk, scan, flags) для проверок.
    fn describe(inputs: &[INPUT]) -> Vec<(u16, u16, u32)> {
        inputs
            .iter()
            .map(|input| unsafe {
                let ki = input.Anonymous.ki;
                (ki.wVk, ki.wScan, ki.dwFlags)
            })
            .collect()
    }

    #[test]
    fn empty_text_is_noop() {
        assert!(insert_text_at_cursor("").is_ok());
        assert!(text_to_inputs("").is_empty());
    }

    #[test]
    fn unicode_char_uses_scan_code() {
        // Кириллический символ уходит как Unicode-скан-код без виртуальной клавиши.
        let events = describe(&text_to_inputs("А"));
        assert_eq!(
            events,
            vec![
                (0, 'А' as u16, KEYEVENTF_UNICODE),
                (0, 'А' as u16, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
            ]
        );
    }

    #[test]
    fn newline_variants_produce_single_return() {
        for text in ["\n", "\r\n", "\r"] {
            let events = describe(&text_to_inputs(text));
            assert_eq!(
                events,
                vec![(VK_RETURN, 0, 0), (VK_RETURN, 0, KEYEVENTF_KEYUP)],
                "перевод строки {text:?} должен давать ровно одно нажатие Enter"
            );
        }
    }

    #[test]
    fn emoji_is_sent_as_surrogate_pair() {
        // 😀 = U+1F600 = суррогатная пара D83D DE00: два нажатия и два отпускания.
        let events = describe(&text_to_inputs("😀"));
        let scans: Vec<u16> = events.iter().map(|&(_, scan, _)| scan).collect();
        assert_eq!(scans, vec![0xD83D, 0xD83D, 0xDE00, 0xDE00]);
    }
}
