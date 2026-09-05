//! Вставка распознанного текста в приложение, где стоит курсор.
//!
//! Текст помещается в системный буфер обмена, после чего имитируется
//! нажатие Ctrl+V. Windows сама вставляет текст в позицию курсора
//! (или заменяет выделенный текст) активного окна. Такой подход
//! работает с любым приложением: браузером, Word, блокнотом и т.п.

#![cfg(target_os = "windows")]

use std::sync::Mutex;

// Виртуальные коды клавиш и флаги для keybd_event (WinUser.h).
const VK_CONTROL: u8 = 0x11;
const VK_V: u8 = 0x56;
const KEYEVENTF_KEYUP: u32 = 0x0002;

#[link(name = "user32")]
unsafe extern "system" {
    fn keybd_event(b_vk: u8, b_scan: u8, dw_flags: u32, dw_extra_info: usize);
}

/// Буфер обмена защищаем мьютексом: программа может делать вставки из
/// нескольких потоков, но к системному буферу Windows доступ только один.
static CLIPBOARD_LOCK: Mutex<()> = Mutex::new(());

/// Кладёт `text` в буфер обмена и вставляет его в приложение под курсором
/// через Ctrl+V. Windows сама решает: вставить между словами по позиции
/// курсора или заменить выделенный текст.
///
/// Возвращает `Ok(true)` при успехе или сообщение об ошибке.
pub fn insert_text_at_cursor(text: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let _guard = CLIPBOARD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Кладём текст в буфер обмена и закрываем буфер, чтобы владение не держать.
    let mut clipboard = arboard::Clipboard::new()?;
    clipboard.set_text(text.to_string())?;
    drop(clipboard);

    // Небольшая пауза, чтобы приложение успело «увидеть» новый буфер.
    std::thread::sleep(std::time::Duration::from_millis(30));

    // Имитируем Ctrl+V: нажимаем Ctrl, нажимаем V, отпускаем V, отпускаем Ctrl.
    unsafe {
        keybd_event(VK_CONTROL, 0, 0, 0);
        keybd_event(VK_V, 0, 0, 0);
        keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0);
        keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0);
    }

    Ok(true)
}
