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
        Self::acquire_on(GUARD_PORT)
    }

    /// То же, что `acquire`, но на указанном порту. Нужно для тестов: они
    /// занимают свободный временный порт, поэтому детерминированы даже когда
    /// реальное приложение уже запущено (оно держит `GUARD_PORT`).
    pub fn acquire_on(port: u16) -> Result<Self, String> {
        let address = format!("127.0.0.1:{port}");
        match TcpListener::bind(&address) {
            Ok(listener) => Ok(Self {
                _listener: listener,
            }),
            Err(err) => Err(format!(
                "Приложение уже запущено (порт {port} занят: {err}). \
                 Одна копия VoiceAI уже работает — вторая не нужна."
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener as NetTcpListener;

    /// Возвращает временный свободный порт на loopback.
    fn free_port() -> u16 {
        let probe = NetTcpListener::bind("127.0.0.1:0").expect("выделить временный порт");
        probe.local_addr().expect("адрес пробного порта").port()
    }

    /// Пока первая копия держит порт, вторая не может запуститься;
    /// после закрытия первой порт снова свободен. Написан на временном
    /// порту, чтобы проходить даже при уже запущенном приложении.
    #[test]
    fn second_guard_rejected_until_first_released() {
        let port = free_port();
        let first = SingleInstanceGuard::acquire_on(port).expect("первая копия занимает порт");
        let second = SingleInstanceGuard::acquire_on(port);
        assert!(
            second.is_err(),
            "вторая копия не должна запускаться, пока жива первая"
        );
        drop(first);
        assert!(
            SingleInstanceGuard::acquire_on(port).is_ok(),
            "после закрытия первой копии порт снова доступен"
        );
    }
}
