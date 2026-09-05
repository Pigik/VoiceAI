//! Автозапуск приложения при входе пользователя в систему.
//!
//! Windows: запись в реестре `HKCU\...\Run`. Linux: файл `voiceai.desktop`
//! в `~/.config/autostart`. macOS: LaunchAgent-плейлист в
//! `~/Library/LaunchAgents`. Включается настройкой `auto_start`.

#[cfg(not(target_os = "windows"))]
use std::path::PathBuf;

/// Включает или выключает автозапуск. Возвращает сообщение о результате.
pub fn apply_autostart(enabled: bool) -> Result<String, String> {
    #[cfg(target_os = "windows")]
    {
        windows_autostart(enabled)
    }
    #[cfg(target_os = "linux")]
    {
        xdg_autostart(enabled)
    }
    #[cfg(target_os = "macos")]
    {
        macos_autostart(enabled)
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        let _ = enabled;
        Err("Автозапуск не поддерживается на этой платформе".to_string())
    }
}

/// Добавляет/убирает запись в системном реестре Windows.
#[cfg(target_os = "windows")]
fn windows_autostart(enabled: bool) -> Result<String, String> {
    use winreg::RegKey;
    use winreg::enums::{HKEY_CURRENT_USER, KEY_READ, KEY_WRITE};

    let exe = std::env::current_exe().map_err(|err| err.to_string())?;
    let exe = format!("\"{}\"", exe.to_string_lossy());

    let run = RegKey::predef(HKEY_CURRENT_USER)
        .open_subkey_with_flags(
            "Software\\Microsoft\\Windows\\CurrentVersion\\Run",
            KEY_READ | KEY_WRITE,
        )
        .map_err(|err| format!("Не удалось открыть раздел автозапуска: {err}"))?;

    if enabled {
        run.set_value("VoiceAI", &exe)
            .map_err(|err| format!("Не удалось включить автозапуск: {err}"))?;
        Ok("Автозапуск включён (Windows)".to_string())
    } else {
        match run.delete_value("VoiceAI") {
            Ok(()) => Ok("Автозапуск выключен (Windows)".to_string()),
            // Записи не было — это тоже «выключено», ошибкой не считаем.
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Ok("Автозапуск выключен (Windows)".to_string())
            }
            Err(err) => Err(format!("Не удалось выключить автозапуск: {err}")),
        }
    }
}

/// Создаёт/удаляет .desktop-файл XDG Autostart (Linux).
#[cfg(target_os = "linux")]
fn xdg_autostart(enabled: bool) -> Result<String, String> {
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME").ok().map(|home| {
                let mut home = PathBuf::from(home);
                home.push(".config");
                home
            })
        })
        .ok_or_else(|| "Не найдена папка конфигурации (HOME)".to_string())?;

    let file = base.join("autostart").join("voiceai.desktop");
    if !enabled {
        if file.exists() {
            std::fs::remove_file(&file).map_err(|err| err.to_string())?;
        }
        return Ok("Автозапуск выключен (Linux)".to_string());
    }

    let exe = std::env::current_exe().map_err(|err| err.to_string())?;
    let content = format!(
        "[Desktop Entry]\nType=Application\nName=VoiceAI\nExec={}\nComment=Диктофон с распознаванием речи\n",
        exe.to_string_lossy()
    );
    std::fs::create_dir_all(file.parent().ok_or("Нет папки для автозапуска")?)
        .map_err(|err| err.to_string())?;
    std::fs::write(&file, content).map_err(|err| format!("Не удалось записать {file:?}: {err}"))?;
    Ok("Автозапуск включён (Linux)".to_string())
}

/// Создаёт/удаляет LaunchAgent (macOS).
#[cfg(target_os = "macos")]
fn macos_autostart(enabled: bool) -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|_| "Не найдена папка HOME".to_string())?;
    let file = PathBuf::from(&home)
        .join("Library")
        .join("LaunchAgents")
        .join("com.voiceai.autostart.plist");

    if !enabled {
        if file.exists() {
            std::fs::remove_file(&file).map_err(|err| err.to_string())?;
        }
        return Ok("Автозапуск выключен (macOS)".to_string());
    }

    let exe = std::env::current_exe().map_err(|err| err.to_string())?;
    let content = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
         \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\"><dict>\
         <key>Label</key><string>com.voiceai.autostart</string>\
         <key>ProgramArguments</key><array><string>{}</string></array>\
         <key>RunAtLoad</key><true/>\
         </dict></plist>\n",
        exe.to_string_lossy()
    );
    std::fs::create_dir_all(file.parent().ok_or("Нет папки для автозапуска")?)
        .map_err(|err| err.to_string())?;
    std::fs::write(&file, content).map_err(|err| format!("Не удалось записать {file:?}: {err}"))?;
    Ok("Автозапуск включён (macOS)".to_string())
}
