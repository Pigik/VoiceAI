//! Защита от запуска второй копии приложения.
//!
//! Работает через занятие локального TCP-порта-«метки»: вторая копия не
//! сможет его занять и завершится, не трогая первую. Когда первая копия
//! закрывается, порт освобождается и запуск снова становится возможен.

use std::net::TcpListener;

/// Локальный порт-«метка» процесса. Число выбрано из нечасто используемого
/// диапазона, чтобы не пересекаться с другими программами.
const GUARD_PORT: u16 = 43487;

/// Живой «замок» одной копии приложения. Пока объект жив, порт занят.
pub struct SingleInstanceGuard {
    _listener: TcpListener,
}

impl SingleInstanceGuard {
    /// Пытается занять порт-«метку». При неудаче возвращает понятное
    /// сообщение о том, что приложение уже запущено.
    pub fn acquire() -> Result<Self, String> {
        let address = format!("127.0.0.1:{GUARD_PORT}");
        match TcpListener::bind(&address) {
            Ok(listener) => Ok(Self {
                _listener: listener,
            }),
            Err(err) => Err(format!(
                "Приложение уже запущено (порт {GUARD_PORT} занят: {err}). \
                 Одна копия VoiceAI уже работает — вторая не нужна."
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Пока первая копия держит порт, вторая не может запуститься;
    /// после закрытия первой порт снова свободен.
    #[test]
    fn second_guard_rejected_until_first_released() {
        let first = SingleInstanceGuard::acquire().expect("первая копия занимает порт");
        let second = SingleInstanceGuard::acquire();
        assert!(
            second.is_err(),
            "вторая копия не должна запускаться, пока жива первая"
        );
        drop(first);
        assert!(
            SingleInstanceGuard::acquire().is_ok(),
            "после закрытия первой копии порт снова доступен"
        );
    }
}
