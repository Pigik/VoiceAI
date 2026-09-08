//! Иконка приложения в системном трее (Windows).
//!
//! Иконка создаётся на главном потоке (внутри цикла eframe), а её вид и
//! подсказка отражают состояние записи: синий кружок — ожидание, красный —
//! идёт запись. Контекстное меню иконки возвращает скрытое окно на экран
//! («Открыть окно») или полностью завершает приложение («Выход») — команды
//! разбирает фоновый поток `spawn_tray_event_handler` в main.rs.

#![cfg(target_os = "windows")]

use tray_icon::menu::{Menu, MenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

/// Размер стороны иконки в пикселях (трей рисует её сам в нужном масштабе).
const ICON_SIZE: u32 = 32;

/// Идентификатор пункта меню «Открыть окно» — по нему обработчик событий
/// трея в main.rs отличает команды меню друг от друга.
pub const MENU_OPEN_ID: &str = "voiceai-tray-open";
/// Идентификатор пункта меню «Выход».
pub const MENU_QUIT_ID: &str = "voiceai-tray-quit";

/// Собирает контекстное меню иконки: вернуть окно из фона или выйти.
fn build_menu() -> Result<Menu, String> {
    let open = MenuItem::with_id(MENU_OPEN_ID, "Открыть окно", true, None);
    let quit = MenuItem::with_id(MENU_QUIT_ID, "Выход", true, None);
    let menu = Menu::new();
    menu.append(&open)
        .map_err(|err| format!("Не удалось добавить пункт меню: {err}"))?;
    menu.append(&quit)
        .map_err(|err| format!("Не удалось добавить пункт меню: {err}"))?;
    Ok(menu)
}

/// Рисует «микрофон»-кружок: синий в ожидании, красный во время записи.
fn build_icon(recording: bool) -> Icon {
    let (r, g, b) = if recording {
        (239, 68, 68)
    } else {
        (59, 130, 246)
    };
    let mut rgba = vec![0u8; (ICON_SIZE * ICON_SIZE * 4) as usize];

    let center = (ICON_SIZE - 1) as f32 / 2.0;
    let radius = (ICON_SIZE as f32 - 3.0) / 2.0;
    let inner = radius - 2.75;

    for y in 0..ICON_SIZE {
        for x in 0..ICON_SIZE {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            let index = ((y * ICON_SIZE + x) * 4) as usize;
            if dist <= inner {
                rgba[index] = r;
                rgba[index + 1] = g;
                rgba[index + 2] = b;
                rgba[index + 3] = 255;
            } else if dist <= radius && recording {
                // Во время записи — светлое кольцо вокруг красного кружка.
                rgba[index] = 255;
                rgba[index + 1] = 255;
                rgba[index + 2] = 255;
                rgba[index + 3] = 255;
            }
        }
    }

    Icon::from_rgba(rgba, ICON_SIZE, ICON_SIZE).expect("иконка трея имеет валидный размер")
}

/// Создаёт иконку в системном трее с текущим состоянием записи и меню.
pub fn create_tray(recording: bool) -> Result<TrayIcon, String> {
    let menu = build_menu()?;
    let tooltip = if recording {
        "VoiceAI — идёт запись"
    } else {
        "VoiceAI — готов к записи"
    };
    TrayIconBuilder::new()
        .with_tooltip(tooltip)
        .with_icon(build_icon(recording))
        .with_menu(Box::new(menu))
        .build()
        .map_err(|err| format!("Не удалось создать иконку в трее: {err}"))
}

/// Обновляет вид и подсказку иконки под текущее состояние записи.
pub fn update_tray(tray: &TrayIcon, recording: bool) {
    let tooltip = if recording {
        "VoiceAI — идёт запись"
    } else {
        "VoiceAI — готов к записи"
    };
    let _ = tray.set_icon(Some(build_icon(recording)));
    let _ = tray.set_tooltip(Some(tooltip));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Сгенерированная иконка собирается и в режиме ожидания, и в записи
    /// (паника внутри `build_icon` означала бы битый буфер RGBA).
    #[test]
    fn icons_build_for_both_states() {
        for recording in [false, true] {
            let _icon = build_icon(recording);
        }
    }
}
