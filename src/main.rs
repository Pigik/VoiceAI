// На Windows скрываем консольное окно: приложение запускается только как GUI.
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

#[cfg(target_os = "windows")]
use voiceai::insert;
use voiceai::stats_logger::{ReplicaLog, log_replica};
use voiceai::{
    analytics, audio, autostart, download, models, postprocess, settings, single_instance,
    stats_logger, whisper_config,
};

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use global_hotkey::hotkey::{Code, HotKey};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::onnx::gigaam::GigaAMModel;
use transcribe_rs::{SpeechModel, TranscribeOptions};
use voiceai::settings::{PressMode, Settings, SpeechEngine};
use whisper_rs::{WhisperContext, WhisperContextParameters};

/// Файл последней записи (в рабочей папке программы); расшифровка
/// сохраняется рядом с ним в `output.txt`.
const OUTPUT_FILE: &str = "output.wav";
/// Файл журнала событий (рядом с записью). Туда пишутся все важные шаги
/// работы: сохранение, расшифровка, ошибки. Помогает понять, что пошло не так,
/// когда окно программы не показывает деталей.
const LOG_FILE: &str = "voiceai.log";
/// Размер журнала, после которого он переименовывается в `voiceai.log.old`
/// и начинается заново (чтобы файл не рос бесконечно).
const LOG_MAX_BYTES: u64 = 5 * 1024 * 1024;
/// Имя файла модели Whisper (GGML .bin). Модель будет искаться рядом
/// с программой, в текущей папке, в папке models/ или по переменной
/// окружения WHISPER_MODEL (путь из настроек имеет приоритет).
const MODEL_FILE: &str = "ggml-large-v3-turbo.bin";

/// Что делать с аргументами командной строки.
enum CliAction {
    /// Обычный запуск: графическое окно.
    Run,
    /// Вывести версию и выйти с кодом 0.
    PrintVersion,
    /// Вывести справку и выйти с кодом 0.
    PrintHelp,
}

/// Разбирает аргументы командной строки. Первый же «флаг действия»
/// переопределяет запуск GUI: версия/справка важнее прочих аргументов.
fn parse_cli(args: &[String]) -> CliAction {
    for arg in args {
        match arg.as_str() {
            "-V" | "--version" => return CliAction::PrintVersion,
            "-h" | "--help" => return CliAction::PrintHelp,
            _ => {}
        }
    }
    CliAction::Run
}

/// Подключает вывод к консоли, из которой запущена программа.
///
/// На Windows программа собрана как графическая (без своей консоли), поэтому
/// без этого `--help` и `--version` ничего бы не печатали. Командная строка
/// не ждёт графические программы, поэтому текст появляется после приглашения.
#[cfg(target_os = "windows")]
fn attach_parent_console() {
    use windows_sys::Win32::System::Console::{ATTACH_PARENT_PROCESS, AttachConsole};
    unsafe {
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

/// Печатает версию программы (значение из Cargo.toml).
fn print_version() {
    println!("VoiceAI {}", env!("CARGO_PKG_VERSION"));
}

/// Печатает справку по флагам.
fn print_help() {
    println!(
        "VoiceAI {} — диктофон с локальным распознаванием речи (Whisper).",
        env!("CARGO_PKG_VERSION")
    );
    println!();
    println!("Использование: VoiceAI [ФЛАГИ]");
    println!();
    println!("Флаги:");
    println!("  -h, --help      показать эту справку");
    println!("  -V, --version   показать версию программы");
    println!();
    println!(
        "Без флагов открывается графическое окно. Зажмите клавишу записи (по умолчанию F1), \
         говорите, отпустите — текст появится под курсором."
    );
}

/// Общее состояние приложения, которым обмениваются поток диктофона и GUI.
#[derive(Default)]
struct AppState {
    /// Идёт ли сейчас запись (в зависимости от режима клавиши).
    recording: bool,
    /// Последнее короткое сообщение для показа под заголовком (например, файл сохранён).
    status: String,
    /// Текущие настройки пользователя.
    settings: Settings,
    /// Имена доступных устройств ввода (заполняются при старте и обновляются).
    devices: Vec<String>,
    /// Текущий уровень входного сигнала в диапазоне 0..=1 (для индикатора).
    input_level: f32,
    /// Журнал реплик (записей) за текущую сессию — для окна «Журнал».
    replicas: Vec<ReplicaLog>,
    /// Состояние скачивания второй модели (GigaAM) для окна настроек.
    download: DownloadState,
    /// Пользователь выбрал «Выход» (кнопка в окне или меню трея): следующее
    /// закрытие окна завершает приложение, а не прячет его в трей.
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    want_exit: bool,
    /// Окно скрыто в фоновый режим (трей) — приложение продолжает работать.
    #[cfg_attr(not(target_os = "windows"), allow(dead_code))]
    hidden: bool,
}

/// Состояние скачивания модели GigaAM: идёт ли загрузка и её прогресс.
#[derive(Clone, Default)]
struct DownloadState {
    /// Идёт ли скачивание прямо сейчас.
    active: bool,
    /// Сколько байт уже скачано.
    downloaded: u64,
    /// Сколько байт ожидается всего (0 — размер пока неизвестен).
    total: u64,
}

/// GUI-приложение: белое окно с подсказкой по центру, окна настроек и журнала.
struct DictophoneApp {
    state: Arc<Mutex<AppState>>,
    /// Открыто ли окно настроек.
    show_settings: bool,
    /// Открыто ли окно журнала реплик.
    show_journal: bool,
    /// ID выбранной записи в журнале (для просмотра полного текста).
    selected_replica: Option<uuid::Uuid>,
    /// Куда экспортировать профиль настроек (путь в поле ввода).
    export_path: String,
    /// Откуда импортировать профиль настроек (путь в поле ввода).
    import_path: String,
    /// Идёт ли ожидание нажатия клавиши для переназначения горячей клавиши.
    capture_hotkey: bool,
    /// Менеджер глобальных горячих клавиш (обязан жить на главном потоке,
    /// где работает цикл сообщений окна-приёмника системного хоткея).
    hotkey_manager: Option<GlobalHotKeyManager>,
    /// Какая клавиша сейчас зарегистрирована (сверяется при каждом кадре).
    current_hotkey: Option<HotKey>,
    /// Иконка в системном трее (Windows) и состояние, для которого она нарисована.
    #[cfg(target_os = "windows")]
    tray: Option<tray_icon::TrayIcon>,
    #[cfg(target_os = "windows")]
    tray_recording: Option<bool>,
}

impl DictophoneApp {
    /// `open_settings` — открыть окно настроек сразу (первый запуск или ошибка конфига).
    fn new(
        state: Arc<Mutex<AppState>>,
        open_settings: bool,
        hotkey_manager: Option<GlobalHotKeyManager>,
    ) -> Self {
        let state_settings = state.lock().map(|s| s.settings.clone()).unwrap_or_default();
        Self {
            state,
            show_settings: open_settings,
            show_journal: false,
            selected_replica: None,
            export_path: default_profile_path(),
            import_path: String::new(),
            capture_hotkey: false,
            current_hotkey: Some(hotkey_from_settings(&state_settings)),
            hotkey_manager,
            #[cfg(target_os = "windows")]
            tray: None,
            #[cfg(target_os = "windows")]
            tray_recording: None,
        }
    }
}

/// Путь к файлу профиля по умолчанию (в рабочей папке, рядом с output.wav).
fn default_profile_path() -> String {
    std::env::current_dir()
        .map(|dir| {
            dir.join("voiceai-profile.json")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|_| "voiceai-profile.json".to_string())
}

impl eframe::App for DictophoneApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Обновляем окно даже когда ничего не меняется, чтобы показывать статус.
        ui.ctx().request_repaint_after(Duration::from_millis(50));

        // Закрытие окна крестиком (Windows) не завершает приложение, а прячет его
        // в трей: запись и горячая клавиша продолжают работать. Полный выход — через кнопку
        // «Выход» в правом верхнем углу или меню иконки в трее.
        #[cfg(target_os = "windows")]
        if ui.ctx().input(|i| i.viewport().close_requested()) {
            let exit_chosen = self.state.lock().map(|s| s.want_exit).unwrap_or(false);
            if !exit_chosen {
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::CancelClose);
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::Visible(false));
                let was_hidden = self.state.lock().map(|s| s.hidden).unwrap_or(false);
                if let Ok(mut s) = self.state.lock() {
                    s.hidden = true;
                }
                if !was_hidden {
                    append_log(
                        &self.state,
                        "Окно скрыто в трей: приложение продолжает работать в фоне, \
                         клавиша записи активна. Вернуть окно — иконка в трее, выйти — \
                         меню трея или кнопка «Выход»."
                            .to_string(),
                    );
                }
            }
        }

        let (recording, status, settings, devices, input_level) = match self.state.lock() {
            Ok(state) => (
                state.recording,
                state.status.clone(),
                state.settings.clone(),
                state.devices.clone(),
                state.input_level,
            ),
            // Мьютекс отравлен паникой другого потока — показываем ошибку.
            Err(_) => {
                egui::CentralPanel::default().show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Ошибка состояния приложения");
                    });
                });
                return;
            }
        };
        let mut new_settings = settings.clone();

        // Переназначаем горячую клавишу на лету: если в настройках сменилась
        // (окно настроек или импорт профиля), снимаем старую и регистрируем новую.
        // Менеджер живёт на главном потоке, поэтому и переназначение тут же.
        if let Some(manager) = &self.hotkey_manager {
            let desired = hotkey_from_settings(&new_settings);
            if self.current_hotkey != Some(desired) {
                if let Some(old) = self.current_hotkey {
                    let _ = manager.unregister(old);
                }
                match manager.register(desired) {
                    Ok(()) => {
                        append_log(
                            &self.state,
                            format!("Горячая клавиша: {}", hotkey_label(&new_settings)),
                        );
                    }
                    Err(err) => {
                        set_status(
                            &self.state,
                            format!(
                                "Не удалось зарегистрировать клавишу {}: {err}",
                                hotkey_label(&new_settings)
                            ),
                        );
                    }
                }
                self.current_hotkey = Some(desired);
            }
        }

        // Иконка в трее (Windows): создаём один раз на главном потоке и
        // перерисовываем под состояние записи, когда оно меняется.
        #[cfg(target_os = "windows")]
        {
            if self.tray.is_none() {
                if self.tray_recording.is_none() {
                    match voiceai::tray::create_tray(recording) {
                        Ok(tray) => {
                            self.tray_recording = Some(recording);
                            self.tray = Some(tray);
                        }
                        Err(err) => {
                            append_log(&self.state, format!("Иконка в трее: {err}"));
                            // Больше не пробуем, чтобы не спамить в журнал.
                            self.tray_recording = Some(recording);
                        }
                    }
                }
            } else if self.tray_recording != Some(recording) {
                if let Some(tray) = self.tray.as_ref() {
                    voiceai::tray::update_tray(tray, recording);
                }
                self.tray_recording = Some(recording);
            }
        }

        // Кнопка «Выход» в правом верхнем углу главного окна (Windows): полное
        // завершение приложения. В отличие от крестика, который скрывает окно
        // в трей, она закрывает приложение с концами — оно исчезает из
        // процессов и памяти.
        #[cfg(target_os = "windows")]
        egui::Area::new(egui::Id::new("voiceai_exit_button"))
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
            .show(ui.ctx(), |ui| {
                if ui.button("Выход").clicked() {
                    if let Ok(mut s) = self.state.lock() {
                        s.want_exit = true;
                    }
                    append_log(&self.state, "Завершение работы приложения...".to_string());
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

        // Кнопка настроек — всегда внизу окна, в отдельной нижней панели,
        // чтобы она гарантированно не перекрывала и не «сдвигала» текст.
        egui::Panel::bottom("settings_panel")
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(8, 6)),
            )
            .show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.horizontal(|ui| {
                        if ui.button("⚙ Настройки").clicked() {
                            self.show_settings = true;
                        }
                        if ui.button("📋 Журнал").clicked() {
                            self.show_journal = true;
                        }
                    });
                });
            });

        // Белый фон всего окна и содержимое по центру оставшейся области.
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::WHITE)
                    .inner_margin(16.0),
            )
            .show(ui, |ui| {
                // Считаем реальную высоту содержимого, чтобы точно отцентровать
                // его по вертикали и не «съезжать» к нижней кромке окна.
                let title = if recording {
                    "Идёт запись...".to_string()
                } else {
                    format!("Используйте {} для записи", hotkey_label(&settings))
                };
                let title_h = ui.text_style_height(&egui::TextStyle::Heading);
                let meter_h = if recording {
                    ui.text_style_height(&egui::TextStyle::Body) + 14.0
                } else {
                    0.0
                };
                let warn_h = if recording && input_level <= 0.0005 {
                    ui.text_style_height(&egui::TextStyle::Body)
                } else {
                    0.0
                };
                let status_h = if status.is_empty() {
                    0.0
                } else {
                    8.0 + ui.text_style_height(&egui::TextStyle::Body)
                };
                let content_h = title_h + meter_h + warn_h + status_h;

                ui.vertical_centered(|ui| {
                    let avail = ui.available_height();
                    if avail > content_h {
                        ui.add_space((avail - content_h) / 2.0);
                    }
                    ui.heading(title);

                    // Индикатор уровня сигнала во время записи.
                    if recording {
                        ui.add_space(10.0);
                        let meter_width = ui.available_width().min(360.0);
                        let bar = egui::ProgressBar::new((input_level * 100.0).clamp(0.0, 100.0))
                            .desired_width(meter_width)
                            .desired_height(12.0);
                        ui.add(bar);
                        // Предупреждение, если сигнала практически нет.
                        if input_level <= 0.0005 {
                            ui.add_space(4.0);
                            ui.colored_label(
                                egui::Color32::from_rgb(200, 0, 0),
                                "Микрофон пишет тишину: проверьте выбор устройства ввода",
                            );
                        }
                    }

                    if !status.is_empty() {
                        ui.add_space(8.0);
                        ui.label(&status);
                    }
                });
            });

        // Окно настроек поверх основного окна.
        // Если окно закрыто (крестиком или сворачиванием), снимаем режим
        // перехвата клавиши, чтобы при повторном открытии настройки не
        // ждали нажатия.
        if !self.show_settings {
            self.capture_hotkey = false;
        }
        if self.show_settings {
            let mut open = true;
            egui::Window::new("Настройки")
                .open(&mut open)
                .collapsible(false)
                .resizable(false)
                .min_width(420.0)
                .default_height(440.0)
                .max_height(440.0)
                .show(ui.ctx(), |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.settings_contents(ui, &mut new_settings, &devices);
                    });
                });
            if !open {
                self.show_settings = false;
            }

            // Применяем изменения настроек, синхронизируем автозапуск и сохраняем.
            if new_settings != settings {
                if new_settings.auto_start != settings.auto_start {
                    match autostart::apply_autostart(new_settings.auto_start) {
                        Ok(message) => append_log(&self.state, message),
                        Err(err) => append_log(&self.state, format!("Автозапуск: {err}")),
                    }
                }
                match self.state.lock() {
                    Ok(mut state) => {
                        state.settings = new_settings.clone();
                    }
                    Err(_) => {
                        append_log(
                            &self.state,
                            "Настройки применены, но не удалось записать в память \
                             (мьютекс отравлен). Перезапустите приложение."
                                .to_string(),
                        );
                    }
                }
                match settings::save_settings(&new_settings) {
                    Ok(()) => {}
                    Err(err) => append_log(&self.state, format!("Сохранение настроек: {err}")),
                }
            }
        }

        // Окно журнала реплик. Журнал клонируется только когда окно открыто.
        if self.show_journal {
            self.journal_window(ui.ctx(), &settings);
        }
    }
}

impl DictophoneApp {
    /// Содержимое окна настроек: правит `new_settings`, а сохраняет их
    /// вызывающий код (если что-то изменилось).
    fn settings_contents(
        &mut self,
        ui: &mut egui::Ui,
        new_settings: &mut Settings,
        devices: &[String],
    ) {
        // Выбор устройства ввода.
        let selected_label = new_settings
            .input_device
            .clone()
            .unwrap_or_else(|| "По умолчанию (авто)".to_string());
        egui::ComboBox::from_label("Устройство ввода")
            .selected_text(selected_label)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(new_settings.input_device.is_none(), "По умолчанию (авто)")
                    .clicked()
                {
                    new_settings.input_device = None;
                }
                for device in devices {
                    let checked = new_settings.input_device.as_deref() == Some(device.as_str());
                    if ui.selectable_label(checked, device).clicked() {
                        new_settings.input_device = Some(device.clone());
                    }
                }
            });

        ui.add_space(10.0);
        // Усиление входного сигнала.
        ui.add(
            egui::Slider::new(&mut new_settings.input_gain, settings::INPUT_GAIN_RANGE)
                .text("Усиление входа")
                .logarithmic(true),
        );
        ui.add_space(2.0);
        ui.label(
            "1.0 — без изменений. Значения ниже 1.0 делают запись тише и ухудшают распознавание",
        );

        ui.add_space(10.0);
        // Шумоподавление.
        ui.checkbox(
            &mut new_settings.noise_reduction,
            "Подавление фонового шума",
        );

        ui.add_space(10.0);
        // Режим приватности: текст реплик не сохраняется в журнале.
        ui.checkbox(
            &mut new_settings.privacy_mode,
            "Режим приватности (не сохранять текст реплик)",
        );
        ui.add_space(2.0);
        ui.label(
            "Статистика и число слов записываются, а текст расшифровки не сохраняется \
             ни в журнале, ни в voiceai.log",
        );

        ui.add_space(10.0);
        // Хранение аудиофайла записи.
        ui.checkbox(
            &mut new_settings.keep_audio,
            "Сохранять аудио записи (output.wav)",
        );
        ui.add_space(2.0);
        ui.label("Выключите, чтобы не оставлять файлы записей — расшифровка всё равно работает");

        ui.add_space(10.0);
        // Клавиша вызова записи (переназначается нажатием).
        ui.label("Клавиша вызова записи:");
        let capture_label = if self.capture_hotkey {
            "Нажмите клавишу (F1–F24)…".to_string()
        } else {
            hotkey_label(new_settings)
        };
        if ui
            .button(format!("{capture_label} — нажмите, чтобы сменить"))
            .clicked()
        {
            self.capture_hotkey = true;
        }
        if self.capture_hotkey {
            let events = ui.ctx().input(|i| i.events.clone());
            let cancel = events.iter().any(|event| {
                matches!(
                    event,
                    egui::Event::Key {
                        key: egui::Key::Escape,
                        pressed: true,
                        ..
                    }
                )
            });
            let chosen = events.iter().find_map(|event| match event {
                egui::Event::Key {
                    key, pressed: true, ..
                } => egui_key_to_hotkey_code(*key),
                _ => None,
            });
            if cancel {
                self.capture_hotkey = false;
            } else if let Some(code) = chosen {
                new_settings.hotkey = code.to_string();
                self.capture_hotkey = false;
            }
        }
        ui.add_space(2.0);
        ui.label(
            "Клавиша действует по всей системе. Для выбора подходят только \
             функциональные клавиши F1–F24 — буквы и цифры перехватывали бы \
             ввод в других окнах",
        );

        ui.add_space(10.0);
        ui.label("Режим нажатия клавиши:");
        ui.radio_value(
            &mut new_settings.press_mode,
            PressMode::PushToTalk,
            "Рация — запись, пока держишь клавишу",
        );
        ui.radio_value(
            &mut new_settings.press_mode,
            PressMode::Toggle,
            "Нажал — включил, нажал — выключил",
        );

        ui.add_space(10.0);
        // Язык распознавания.
        let language_label = settings::language_label(&new_settings.language);
        egui::ComboBox::from_label("Язык распознавания")
            .selected_text(language_label)
            .show_ui(ui, |ui| {
                for (code, label) in settings::LANGUAGE_CHOICES {
                    if ui
                        .selectable_label(new_settings.language == *code, *label)
                        .clicked()
                    {
                        new_settings.language = code.to_string();
                    }
                }
            });
        ui.add_space(2.0);
        ui.label("«Автоопределение» — Whisper сам определит язык (для GigaAM не важно)");

        ui.add_space(10.0);
        ui.separator();
        // Модель распознавания: Whisper (входит в поставку) или GigaAM
        // (пользователь скачивает её сам — см. src/models.rs).
        ui.label("Модель распознавания:");
        ui.radio_value(
            &mut new_settings.speech_engine,
            SpeechEngine::Whisper,
            "Whisper large-v3-turbo (входит в поставку)",
        );
        ui.radio_value(
            &mut new_settings.speech_engine,
            SpeechEngine::GigaAM,
            "GigaAM v3 e2e-ctc (скачивается отдельно)",
        );
        ui.add_space(4.0);

        if new_settings.speech_engine == SpeechEngine::Whisper {
            // Путь к модели Whisper, пусто — автопоиск.
            ui.label("Путь к модели Whisper (необязательно):");
            let mut model_path = new_settings.model_path.clone().unwrap_or_default();
            let mut model_path_changed = ui.text_edit_singleline(&mut model_path).changed();
            ui.horizontal(|ui| {
                if ui.button("Сбросить путь").clicked() {
                    model_path.clear();
                    model_path_changed = true;
                }
                ui.label("Пусто — автоматический поиск (WHISPER_MODEL, папка программы, models/)");
            });
            if model_path_changed {
                let trimmed = model_path.trim().to_string();
                new_settings.model_path = if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                };
            }
        } else {
            // GigaAM: показывает, установлена ли модель, и позволяет скачать её.
            let gigaam_ready = models::resolve_gigaam_dir();
            let download = self
                .state
                .lock()
                .map(|s| s.download.clone())
                .unwrap_or_default();
            match (gigaam_ready, download.active) {
                (Some(dir), _) => {
                    ui.label(format!("Модель установлена: {}", dir.display()));
                    ui.add_space(2.0);
                    ui.label(
                        "Расшифровка пойдёт через GigaAM: русская речь с пунктуацией, \
                         работает на CPU",
                    );
                }
                (None, true) => {
                    let percent = if download.total > 0 {
                        (download.downloaded as f32 / download.total as f32 * 100.0)
                            .clamp(0.0, 100.0)
                    } else {
                        0.0
                    };
                    let progress_text = if download.total > 0 {
                        format!(
                            "Скачивание: {percent:.0}% ({} МБ)",
                            download.downloaded / (1024 * 1024)
                        )
                    } else {
                        format!("Скачивание: {} МБ...", download.downloaded / (1024 * 1024))
                    };
                    ui.add(
                        egui::ProgressBar::new(percent / 100.0)
                            .desired_width(280.0)
                            .text(progress_text),
                    );
                    ui.add_space(2.0);
                    ui.label(
                        "Модель загружается один раз. По завершении распознавание пойдёт \
                         через GigaAM",
                    );
                }
                (None, false) => {
                    if ui
                        .button("Скачать модель GigaAM v3 (~215 МБ, int8)")
                        .clicked()
                    {
                        start_gigaam_download(self.state.clone());
                    }
                    ui.add_space(2.0);
                    ui.label(
                        "Скачается один раз в папку данных приложения. Модель понимает \
                         русскую речь и сама расставляет пунктуацию",
                    );
                }
            }
        }

        ui.add_space(10.0);
        // Автозапуск при входе в систему.
        let mut auto_start = new_settings.auto_start;
        let auto_start_changed = ui
            .checkbox(&mut auto_start, "Автозапуск при входе в систему")
            .changed();
        if auto_start_changed {
            new_settings.auto_start = auto_start;
        }

        ui.add_space(10.0);
        ui.checkbox(
            &mut new_settings.auto_punctuation,
            "Автоматическая пунктуация",
        );
        ui.add_space(2.0);
        ui.label(
            "Точки, заглавные буквы и абзацы. Если выключить — текст вставляется \
             как распознан",
        );

        ui.add_space(10.0);
        ui.separator();
        // Импорт/экспорт профиля настроек.
        ui.horizontal(|ui| {
            ui.label("Экспорт профиля:");
            ui.text_edit_singleline(&mut self.export_path);
            if ui.button("Экспорт").clicked() {
                match settings::export_settings(&self.export_path, new_settings) {
                    Ok(()) => append_log(
                        &self.state,
                        format!("Профиль настроек сохранён: {}", self.export_path),
                    ),
                    Err(err) => append_log(&self.state, format!("Экспорт профиля: {err}")),
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Импорт профиля:");
            ui.add_sized([250.0, 20.0], |ui: &mut egui::Ui| {
                ui.text_edit_singleline(&mut self.import_path)
            });
        });
        ui.horizontal(|ui| {
            if ui.button("Импорт из файла").clicked() {
                match settings::import_settings(&self.import_path) {
                    Ok(imported) => {
                        *new_settings = imported;
                        append_log(
                            &self.state,
                            format!("Профиль настроек импортирован: {}", self.import_path),
                        );
                    }
                    Err(err) => append_log(&self.state, format!("Импорт профиля: {err}")),
                }
            }
            if ui.button("Очистить поле").clicked() {
                self.import_path.clear();
            }
        });

        ui.add_space(10.0);
        ui.separator();
        // Сброс настроек к значениям по умолчанию.
        if ui
            .button("Сбросить все настройки к значениям по умолчанию")
            .clicked()
        {
            *new_settings = Settings::default();
        }
    }

    /// Окно журнала реплик: список записей, сводка за сегодня, удаление.
    fn journal_window(&mut self, ctx: &egui::Context, settings: &Settings) {
        let replicas = self
            .state
            .lock()
            .map(|s| s.replicas.clone())
            .unwrap_or_default();
        let mut open = true;
        // Пользователь нажал свою кнопку «закрыть» в собственной шапке окна.
        let mut close_requested = false;
        // Действия, выбранные кнопками в этом кадре; применяем после отрисовки,
        // чтобы не менять список во время итерации.
        let mut clear_all = false;
        let mut delete_id: Option<uuid::Uuid> = None;

        egui::Window::new("Журнал реплик")
            .id(egui::Id::new("journal_window"))
            .open(&mut open)
            .collapsible(false)
            // Собственная «шапка» вместо дефолтной: в ней нет лишних кнопок/квадратов
            // и можно поставить нужные действия (очистить журнал, закрыть).
            .title_bar(false)
            .resizable(true)
            .min_width(500.0)
            .default_size([620.0, 560.0])
            .show(ctx, |ui| {
                // --- Шапка окна: заголовок слева, кнопки действий справа. ---
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("Журнал реплик").strong().size(17.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✖").on_hover_text("Закрыть журнал").clicked() {
                            close_requested = true;
                        }
                        if !replicas.is_empty()
                            && ui
                                .button("🗑 Очистить журнал")
                                .on_hover_text("Удалить все записи из журнала")
                                .clicked()
                        {
                            clear_all = true;
                        }
                    });
                });
                ui.add_space(4.0);
                ui.separator();

                if replicas.is_empty() {
                    ui.add_space(16.0);
                    ui.centered_and_justified(|ui| {
                        ui.label(format!(
                            "Записей ещё нет.\nЗажмите {} и поговорите, чтобы они появились здесь.",
                            hotkey_label(settings)
                        ));
                    });
                    return;
                }

                // Сводка за сегодня (аналитика).
                let daily = analytics::calculate_daily_stats(&replicas);
                ui.horizontal(|ui| {
                    ui.label(format!("Записей сегодня: {}", daily.total_replicas));
                    ui.separator();
                    ui.label(format!("Слов: {}", daily.total_words));
                    ui.separator();
                    ui.label(format!(
                        "Средняя длительность: {:.1} с",
                        daily.avg_duration_ms / 1000.0
                    ));
                });
                ui.add_space(6.0);

                // Список записей (новые сверху). Полный текст выбранной записи
                // раскрывается прямо в списке, так что прокрутка сохраняется,
                // а длинные тексты видны целиком.
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for replica in replicas.iter().rev() {
                            let is_selected = self.selected_replica == Some(replica.id);

                            // Карточка записи: клик по ней выделяет/раскрывает текст.
                            let card_hover = ui.scope_builder(
                                egui::UiBuilder::new()
                                    .id_salt(("journal_card", replica.id))
                                    .sense(egui::Sense::click()),
                                |ui| {
                                    let fill = if is_selected {
                                        ui.visuals().widgets.active.bg_fill
                                    } else {
                                        ui.visuals().widgets.inactive.bg_fill
                                    };
                                    egui::Frame::NONE
                                        .inner_margin(egui::Margin::symmetric(8, 6))
                                        .corner_radius(6)
                                        .fill(fill)
                                        .show(ui, |ui| {
                                            ui.set_min_width(ui.available_width());
                                            entry_header_row(
                                                ui,
                                                replica,
                                                is_selected,
                                                &mut delete_id,
                                            );

                                            // Текст записи: превью свёрнуто, полный текст — выбран.
                                            let text = replica.text.as_deref();
                                            if is_selected {
                                                if let Some(text) = text {
                                                    egui::Frame::NONE
                                                        .inner_margin(egui::Margin::symmetric(4, 4))
                                                        .corner_radius(4)
                                                        .fill(ui.visuals().extreme_bg_color)
                                                        .show(ui, |ui| {
                                                            ui.set_min_width(ui.available_width());
                                                            ui.add(
                                                                egui::TextEdit::multiline(
                                                                    &mut text.to_owned(),
                                                                )
                                                                .frame(egui::Frame::NONE)
                                                                .desired_width(f32::INFINITY)
                                                                .interactive(false),
                                                            );
                                                        });
                                                } else {
                                                    ui.label("Текст скрыт (режим приватности).");
                                                }
                                            } else if let Some(text) = text {
                                                let text = text.trim();
                                                if !text.is_empty() {
                                                    let capped = cap_long_words(text, 60);
                                                    let preview: String =
                                                        if capped.chars().count() > 140 {
                                                            let mut cut: String =
                                                                capped.chars().take(140).collect();
                                                            cut.push('…');
                                                            cut
                                                        } else {
                                                            capped
                                                        };
                                                    ui.label(
                                                        egui::RichText::new(preview)
                                                            .color(ui.visuals().weak_text_color()),
                                                    );
                                                }
                                            }
                                        });
                                },
                            );
                            // Клик в свободное место карточки — выбрать/раскрыть запись.
                            if card_hover.response.clicked() && !is_selected {
                                self.selected_replica = Some(replica.id);
                            } else if card_hover.response.clicked() && is_selected {
                                // Повторный клик по открытой записи — свернуть её.
                                self.selected_replica = None;
                            }
                            ui.add_space(6.0);
                        }
                    });
            });

        // Пользователь закрыл окно крестиком (своим или системным управлением).
        if !open || close_requested {
            self.show_journal = false;
            self.selected_replica = None;
        }

        // Применяем удаление записей после отрисовки окна.
        if clear_all || delete_id.is_some() {
            let remaining: Vec<ReplicaLog> = match delete_id {
                Some(id) if !clear_all => replicas.iter().filter(|r| r.id != id).cloned().collect(),
                // Полная очистка журнала одной кнопкой.
                _ => Vec::new(),
            };
            if let Ok(mut s) = self.state.lock() {
                s.replicas = remaining.clone();
            }
            if let Some(path) = replica_log_path() {
                if remaining.is_empty() {
                    stats_logger::clear_replicas(&path);
                } else {
                    stats_logger::rewrite_replicas(&path, &remaining);
                }
            }
            if clear_all || delete_id == self.selected_replica {
                self.selected_replica = None;
            }
        }
    }
}

/// Отрисовывает шапку карточки записи: метаданные слева, кнопки
/// «копировать» и «удалить» справа.
fn entry_header_row(
    ui: &mut egui::Ui,
    replica: &ReplicaLog,
    is_selected: bool,
    delete_id: &mut Option<uuid::Uuid>,
) {
    let when = replica.timestamp.format("%d.%m %H:%M");
    let app = if replica.app_name.trim().is_empty() {
        " (неизвестное окно)".to_string()
    } else {
        format!(" → {}", replica.app_name)
    };
    let duration = replica.duration_ms;

    ui.horizontal(|ui| {
        let head = if is_selected {
            format!(
                "[{when}] {words} слов · {ms} мс · {sec:.1} с{app}",
                words = replica.word_count,
                ms = duration,
                sec = duration as f64 / 1000.0
            )
        } else {
            format!(
                "[{when}] {words} слов · {sec:.1} с{app}",
                words = replica.word_count,
                sec = duration as f64 / 1000.0
            )
        };

        // Метаданные слева. Ширина ограничена, чтобы длинное имя окна
        // (например, путь к файлу) не растягивало карточку и не выталкивало
        // кнопки за правый край окна; полный текст показывается по наведению.
        let reserved_buttons = 70.0;
        let meta_width = (ui.available_width() - reserved_buttons).max(120.0);
        let full_head = head.clone();
        ui.add_sized(
            egui::vec2(meta_width, ui.available_height()),
            egui::Label::new(head).truncate(),
        )
        .on_hover_text(full_head);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            match replica.text.as_deref() {
                Some(text) => {
                    if ui
                        .small_button("📋")
                        .on_hover_text("Скопировать текст записи")
                        .clicked()
                    {
                        ui.ctx().copy_text(text.to_owned());
                    }
                }
                None => {
                    ui.add_enabled(false, egui::Button::new("📋"))
                        .on_hover_text("Текст скрыт (режим приватности)");
                }
            }
            if ui
                .small_button("🗑")
                .on_hover_text("Удалить эту запись")
                .clicked()
            {
                *delete_id = Some(replica.id);
            }
        });
    });
}

/// Укорачивает «монолитные» слова (URL, длинные пути/имена) до `max_word`
/// символов: в режиме переноса по словам такое слово иначе раздувает
/// карточку шире окна и выталкивает кнопки за экран.
fn cap_long_words(text: &str, max_word: usize) -> String {
    text.split_whitespace()
        .map(|word| {
            if word.chars().count() > max_word {
                let mut cut: String = word.chars().take(max_word).collect();
                cut.push('…');
                cut
            } else {
                word.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Команды, которыми поток горячих клавиш общается с диктофоном.
enum KeyCommand {
    /// Клавиша записи нажата — начинаем копить звук.
    Start,
    /// Клавиша записи отпущена — сохраняем накопленный звук в файл.
    Stop,
}

/// Задание на транскрибацию: сэмплы записи и параметры обработки.
struct TranscriptionJob {
    /// Путь к WAV-файлу записи (рядом с ним создастся .txt). Если
    /// `keep_audio == false`, самого файла нет — текст просто ложится
    /// рядом по этому пути.
    wav_path: String,
    /// Накопленные моно-сэмплы записи (родная частота микрофона).
    samples: Vec<i16>,
    /// Частота дискретизации, соответствующая сэмплам.
    sample_rate: u32,
    /// Применять ли автоматическую пунктуацию к распознанному тексту.
    auto_punctuation: bool,
    /// Применять ли подавление фонового шума при подготовке аудио.
    noise_reduction: bool,
    /// Сохранён ли аудиофайл записи (false — расшифровка «из памяти»).
    keep_audio: bool,
}

/// Результат расшифровки с диагностикой для журнала.
///
/// Помимо самого текста возвращаем, что «услышала» модель до пост-обработки,
/// длительность записи и уровень сигнала: по этим данным в `voiceai.log`
/// всегда видно, была ли запись тихой/обрезанной или сбился сам Whisper.
struct TranscribeOutcome {
    /// Путь к сохранённому .txt.
    txt_path: String,
    /// Число слов в итоговом тексте (0 — речи не распознано).
    word_count: usize,
    /// Итоговый текст после пост-обработки (тот, что вставляется).
    text: String,
    /// «Сырой» текст из Whisper до пост-обработки.
    raw_text: String,
    /// Длительность аудио (после подготовки), которое слушал Whisper, в секундах.
    audio_secs: f64,
    /// Максимальный уровень сигнала (0..=1) после подготовки аудио.
    signal_level: f32,
}

/// Пишет пустой .txt и возвращает результат «речи не распознано»
/// (пустой текст и 0 слов): вставка не происходит.
fn no_speech_outcome(
    wav_path: &str,
    audio: &[f32],
) -> Result<TranscribeOutcome, Box<dyn std::error::Error>> {
    let txt_path = replace_wav_extension(wav_path, "txt");
    write_text_file(&txt_path, "")?;
    Ok(TranscribeOutcome {
        txt_path,
        word_count: 0,
        text: String::new(),
        raw_text: String::new(),
        audio_secs: audio.len() as f64 / audio::TARGET_RATE,
        signal_level: peak_amplitude(audio),
    })
}

/// Максимальный абсолютный сэмпл клипа (для журнала диагностики).
fn peak_amplitude(audio: &[f32]) -> f32 {
    audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()))
}

/// Сырой текст Whisper для журнала: пустую строку показываем прочерком.
fn raw_display(raw: &str) -> &str {
    if raw.trim().is_empty() {
        "—"
    } else {
        raw.trim()
    }
}

/// Путь к файлу журнала реплик (jsonl) в папке данных приложения.
fn replica_log_path() -> Option<std::path::PathBuf> {
    settings::config_dir().map(|dir| dir.join("replicas.jsonl"))
}

/// Сохраняет результат расшифровки в журнал реплик: в память (для окна
/// «Журнал») и на диск (переживает перезапуск). Режим приватности
/// убирает текст реплик.
fn record_replica(state: &Arc<Mutex<AppState>>, outcome: &TranscribeOutcome, privacy_mode: bool) {
    let replica = ReplicaLog {
        id: uuid::Uuid::new_v4(),
        timestamp: chrono::Local::now(),
        app_name: active_window_title(),
        duration_ms: (outcome.audio_secs * 1000.0) as u64,
        word_count: outcome.word_count,
        text: (outcome.word_count > 0 && !privacy_mode).then(|| outcome.text.clone()),
    };

    // В память — для мгновенного показа в окне «Журнал».
    if let Ok(mut s) = state.lock() {
        s.replicas.push(replica.clone());
        if s.replicas.len() > stats_logger::MAX_REPLICAS {
            let drop_count = s.replicas.len() - stats_logger::MAX_REPLICAS;
            s.replicas.drain(0..drop_count);
        }
    }

    // На диск — чтобы реплики пережили перезапуск приложения.
    if let Some(path) = replica_log_path() {
        log_replica(&path, replica, privacy_mode);
    }
}

/// Имя активного окна (куда будет вставлен текст). На Windows — заголовок
/// foreground-окна; на других платформах пустая строка.
#[cfg(target_os = "windows")]
fn active_window_title() -> String {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW,
    };
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.is_null() {
            return String::new();
        }
        let len = GetWindowTextLengthW(hwnd);
        if len == 0 {
            return String::new();
        }
        let mut buf = vec![0u16; len as usize + 1];
        let written = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        buf.truncate(written.max(0) as usize);
        String::from_utf16_lossy(&buf)
    }
}

#[cfg(not(target_os = "windows"))]
fn active_window_title() -> String {
    String::new()
}

fn main() -> eframe::Result {
    // Командная строка: --version / --help работают без GUI.
    let args: Vec<String> = std::env::args().skip(1).collect();
    match parse_cli(&args) {
        CliAction::PrintVersion => {
            #[cfg(target_os = "windows")]
            attach_parent_console();
            print_version();
            std::process::exit(0);
        }
        CliAction::PrintHelp => {
            #[cfg(target_os = "windows")]
            attach_parent_console();
            print_help();
            std::process::exit(0);
        }
        CliAction::Run => {}
    }

    // output.wav, output.txt и voiceai.log пишутся в рабочую папку. При
    // автозапуске она бывает системной и закрытой на запись — тогда
    // переходим в папку программы (или в папку данных приложения).
    ensure_writable_workdir();

    // Одна копия приложения: вторая копия не должна мешать первой.
    let _guard = match single_instance::SingleInstanceGuard::acquire() {
        Ok(guard) => guard,
        Err(message) => {
            // Вторая копия не запускается: первую не трогаем, просто выходим.
            write_plain_log(&message);
            #[cfg(target_os = "windows")]
            show_message_box("VoiceAI", &message);
            std::process::exit(0);
        }
    };

    // Загружаем настройки и фиксируем обстоятельства: первый запуск
    // либо понятная ошибка в файле настроек.
    let loaded = settings::load_settings();
    let mut settings = loaded.settings;
    let mut initial_status = if let Some(error) = &loaded.error {
        format!("Настройки: {error}")
    } else if loaded.first_launch {
        "Первый запуск — настройки открываются автоматически".to_string()
    } else {
        String::new()
    };

    // Собираем список устройств ввода и валидируем сохранённый выбор:
    // если устройство больше не подключено — возвращаем по умолчанию.
    let devices = list_input_devices();
    if let Some(name) = &settings.input_device
        && !devices.contains(name)
    {
        settings.input_device = None;
    }

    // Первый запуск: файл настроек создаётся прямо сейчас.
    if loaded.first_launch && settings::save_settings(&settings).is_err() {
        initial_status = "Не удалось создать файл настроек".to_string();
    }

    // Синхронизируем автозапуск системы с настройкой: если пользователь
    // включил его в конфиге, а в системе записи нет — добавляем её.
    if settings.auto_start
        && let Err(err) = autostart::apply_autostart(true)
    {
        initial_status = format!("Автозапуск: {err}");
    }

    let mut app_state = AppState {
        devices,
        settings: settings.clone(),
        status: initial_status,
        ..Default::default()
    };
    // Журнал реплик переживает перезапуск: подгружаем сохранённые записи.
    if let Some(path) = replica_log_path() {
        app_state.replicas = stats_logger::read_replicas(&path);
    }
    let state = Arc::new(Mutex::new(app_state));

    // Канал команд от горячей клавиши к диктофону.
    let (key_tx, key_rx) = mpsc::channel::<KeyCommand>();

    // Регистрируем глобальный хоткей (работает независимо от активного окна).
    let hotkey_manager = create_hotkey_manager(&settings, &state, key_tx);

    // Запускаем диктофон в отдельном потоке, чтобы GUI оставался отзывчивым.
    let recorder_state = state.clone();
    thread::spawn(move || run_recorder(recorder_state, key_rx));

    // Первый запуск или битый конфиг — открываем окно настроек сразу.
    let open_settings = loaded.first_launch || loaded.error.is_some();

    // Создаём окно приложения.
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([480.0, 300.0])
            .with_resizable(false)
            .with_title("Диктофон"),
        ..Default::default()
    };

    let app_state_for_frame = state.clone();
    eframe::run_native(
        "Диктофон",
        options,
        Box::new(move |cc| {
            // Обработчик событий трея живёт в отдельном потоке и возвращает
            // окно из фона/завершает приложение, даже когда окно скрыто.
            #[cfg(target_os = "windows")]
            spawn_tray_event_handler(cc.egui_ctx.clone(), state.clone());
            #[cfg(not(target_os = "windows"))]
            let _ = &cc;
            Ok(Box::new(DictophoneApp::new(
                app_state_for_frame,
                open_settings,
                hotkey_manager,
            )))
        }),
    )
}

/// Делает рабочей папкой ту, куда можно писать файлы программы.
///
/// Обычно это папка, из которой запущена программа (при запуске двойным
/// щелчком — папка с exe). Если туда писать нельзя (автозапуск из
/// `C:\Windows\System32`, установка в Program Files), переходим в папку
/// программы, а если и она закрыта — в папку данных приложения.
fn ensure_writable_workdir() {
    fn is_writable(dir: &std::path::Path) -> bool {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(LOG_FILE))
            .is_ok()
    }

    if std::env::current_dir().is_ok_and(|dir| is_writable(&dir)) {
        return;
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf));
    let data_dir = settings::config_dir().filter(|dir| std::fs::create_dir_all(dir).is_ok());
    for dir in exe_dir.into_iter().chain(data_dir) {
        if is_writable(&dir) && std::env::set_current_dir(&dir).is_ok() {
            return;
        }
    }
}

/// Возвращает имена всех доступных устройств ввода (без повторов).
fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let mut names: Vec<String> = Vec::new();
    if let Ok(input_devices) = host.input_devices() {
        for (index, device) in input_devices.enumerate() {
            let name = device
                .name()
                .unwrap_or_else(|_| format!("Устройство {}", index + 1));
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

/// Ищет устройство ввода по имени среди доступных на текущем хосте.
fn find_input_device(host: &cpal::Host, name: &str) -> Option<cpal::Device> {
    let devices: Vec<cpal::Device> = host.input_devices().ok()?.collect();
    devices
        .into_iter()
        .find(|device| device.name().ok().as_deref() == Some(name))
}

/// Читает текущие настройки из общего состояния (при сбое — значения по умолчанию).
fn read_settings(state: &Arc<Mutex<AppState>>) -> Settings {
    state.lock().map(|s| s.settings.clone()).unwrap_or_default()
}

/// Строит горячую клавишу из настроек; при битом значении — F1.
fn hotkey_from_settings(settings: &Settings) -> HotKey {
    match settings.hotkey.trim().parse::<HotKey>() {
        Ok(hotkey) => HotKey::new(None, hotkey.key),
        Err(_) => HotKey::new(None, Code::F1),
    }
}

/// Человекочитаемое имя горячей клавиши для интерфейса («F1», «F5»).
fn hotkey_label(settings: &Settings) -> String {
    hotkey_from_settings(settings).into_string()
}

/// Клавиши, которые можно назначить вызовом записи (F1–F24).
///
/// Обычные буквы/цифры не даём назначать: глобальная клавиша перехватывает
/// символ по всей системе и сломала бы набор текста в других приложениях.
fn egui_key_to_hotkey_code(key: egui::Key) -> Option<Code> {
    use egui::Key as K;
    use global_hotkey::hotkey::Code as C;
    Some(match key {
        K::F1 => C::F1,
        K::F2 => C::F2,
        K::F3 => C::F3,
        K::F4 => C::F4,
        K::F5 => C::F5,
        K::F6 => C::F6,
        K::F7 => C::F7,
        K::F8 => C::F8,
        K::F9 => C::F9,
        K::F10 => C::F10,
        K::F11 => C::F11,
        K::F12 => C::F12,
        K::F13 => C::F13,
        K::F14 => C::F14,
        K::F15 => C::F15,
        K::F16 => C::F16,
        K::F17 => C::F17,
        K::F18 => C::F18,
        K::F19 => C::F19,
        K::F20 => C::F20,
        K::F21 => C::F21,
        K::F22 => C::F22,
        K::F23 => C::F23,
        K::F24 => C::F24,
        _ => return None,
    })
}

/// Регистрирует глобальную горячую клавишу и запускает поток,
/// который превращает события нажатия/отпускания в команды диктофона.
///
/// Менеджер создаётся на главном потоке (там же жёстко сидит окно-приёмник
/// системного хоткея), поэтому переназначается из GUI через
/// `DictophoneApp::current_hotkey`. Смена клавиши в настройках применяется
/// на лету, без перезапуска программы.
fn create_hotkey_manager(
    settings: &Settings,
    state: &Arc<Mutex<AppState>>,
    key_tx: mpsc::Sender<KeyCommand>,
) -> Option<GlobalHotKeyManager> {
    let manager = match GlobalHotKeyManager::new() {
        Ok(manager) => manager,
        Err(err) => {
            set_status(state, format!("Ошибка горячих клавиш: {err}"));
            return None;
        }
    };

    let hotkey = hotkey_from_settings(settings);
    if let Err(err) = manager.register(hotkey) {
        set_status(
            state,
            format!(
                "Не удалось зарегистрировать клавишу {}: {err}",
                hotkey_label(settings)
            ),
        );
    }

    // Поток, следящий за событиями глобального хоткея. Одновременно
    // зарегистрирована только одна клавиша (старые снимаются перед новыми),
    // поэтому любое пришедшее событие принадлежит текущей клавише.
    thread::spawn(move || {
        loop {
            while let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                let command = match event.state {
                    HotKeyState::Pressed => KeyCommand::Start,
                    HotKeyState::Released => KeyCommand::Stop,
                };
                let _ = key_tx.send(command);
            }
            // Небольшая пауза, чтобы не перегружать процессор.
            thread::sleep(Duration::from_millis(5));
        }
    });

    Some(manager)
}

/// Показывает главное окно приложения (запрос из трея: клик по иконке или
/// пункт «Открыть окно»). Отменяет возможный запрос закрытия, чтобы
/// повторное появление окна не конфликтовало с режимом фона.
#[cfg(target_os = "windows")]
fn show_main_window(ctx: &egui::Context, state: &Arc<Mutex<AppState>>) {
    if let Ok(mut s) = state.lock() {
        s.hidden = false;
    }
    ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
    ctx.request_repaint();
}

/// Запускает полное завершение приложения из трея: ставит флаг выхода и
/// закрывает окно. В следующем кадре обработчик закрытия видит `want_exit`
/// и не прячет окно в трей — приложение завершается полностью.
#[cfg(target_os = "windows")]
fn quit_from_tray(ctx: &egui::Context, state: &Arc<Mutex<AppState>>) {
    if let Ok(mut s) = state.lock() {
        s.want_exit = true;
    }
    append_log(state, "Завершение работы из меню трея...".to_string());
    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    ctx.request_repaint();
}

/// Фоновый поток событий трея.
///
/// Сам иконку не создаёт (её держит главный поток eframe — там работает
/// цикл сообщений окна трея), а только отвечает на события из общих каналов
/// `tray_icon`: клик по иконке или пункт меню «Открыть окно» возвращают окно
/// на экран, пункт «Выход» полностью завершает приложение. Поток живёт всё
/// время работы программы, поэтому команды работают и когда окно скрыто.
#[cfg(target_os = "windows")]
fn spawn_tray_event_handler(ctx: egui::Context, state: Arc<Mutex<AppState>>) {
    use tray_icon::menu::MenuEvent;
    use tray_icon::{MouseButton, TrayIconEvent};

    thread::spawn(move || {
        loop {
            // Клик левой кнопкой по иконке — вернуть главное окно.
            while let Ok(event) = TrayIconEvent::receiver().try_recv() {
                if matches!(
                    event,
                    TrayIconEvent::Click {
                        button: MouseButton::Left,
                        ..
                    }
                ) {
                    show_main_window(&ctx, &state);
                }
            }

            // Команды контекстного меню иконки.
            while let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id == voiceai::tray::MENU_OPEN_ID {
                    show_main_window(&ctx, &state);
                } else if event.id == voiceai::tray::MENU_QUIT_ID {
                    quit_from_tray(&ctx, &state);
                }
            }

            thread::sleep(Duration::from_millis(50));
        }
    });
}

/// Открывает поток захвата, если он ещё не открыт или сменилось устройство.
///
/// Микрофон держим открытым только во время записи: между записями поток
/// закрывается (`capture = None`), и приложение не слушает микрофон.
fn open_capture_if_needed(
    capture: &mut Option<AudioCapture>,
    capture_device: &mut Option<String>,
    device_name: Option<&str>,
    state: &Arc<Mutex<AppState>>,
) {
    let need_reopen = capture.is_none() || capture_device.as_deref() != device_name;
    if !need_reopen {
        return;
    }
    // Старый поток (если был открыт) закроется сам при переприсваивании.
    *capture_device = None;
    match capture_stream(device_name, state.clone()) {
        Ok(new_capture) => {
            // В новом канале уже могут ждать холостые сэмплы — выбрасываем их.
            discard_pending(&new_capture.rx);
            *capture = Some(new_capture);
            *capture_device = device_name.map(String::from);
        }
        Err(err) => {
            *capture = None;
            append_log(state, format!("Ошибка записи: {err}"));
            set_status(state, format!("Не удалось открыть микрофон: {err}"));
        }
    }
}

/// Состояние текущего отрезка записи в цикле диктофона.
struct RecordingSession {
    /// Накопленные моно-сэмплы текущей записи.
    buffer: Vec<i16>,
    /// Идёт ли сейчас запись (набор сэмплов в буфер).
    active: bool,
    /// Момент начала «хвоста» (клавиша отпущена) — дослушиваем остаток.
    tail_start: Option<Instant>,
    /// Экспоненциальная средняя уровня сигнала (для индикатора).
    smoothed_level: f32,
}

/// Начинает запись: открывает микрофон (если нужно) и сбрасывает буфер.
/// Возвращает `false`, если микрофон не удалось открыть — запись не началась.
fn start_recording(
    capture: &mut Option<AudioCapture>,
    capture_device: &mut Option<String>,
    settings: &Settings,
    state: &Arc<Mutex<AppState>>,
    session: &mut RecordingSession,
) -> bool {
    open_capture_if_needed(
        capture,
        capture_device,
        settings.input_device.as_deref(),
        state,
    );
    if capture.is_none() {
        return false;
    }
    // Отбрасываем сэмплы, накопленные до нажатия клавиши.
    if let Some(c) = capture {
        discard_pending(&c.rx);
    }
    session.buffer.clear();
    session.active = true;
    session.tail_start = None;
    session.smoothed_level = 0.0;
    set_level(state, 0.0);
    set_recording(state, true);
    true
}

/// Основной цикл диктофона (режим удержания клавиши или «вкл/выкл»).
///
/// Живёт в отдельном потоке и только обновляет `AppState`, а GUI его считывает.
/// Настройки (устройство ввода, режим клавиши, пунктуация) читает из общего
/// состояния и применяет без перезапуска всей программы. Микрофон открывается
/// только на время записи и закрывается сразу после неё.
fn run_recorder(state: Arc<Mutex<AppState>>, key_rx: Receiver<KeyCommand>) {
    // Канал заданий на транскрибацию. Транскрибатор живёт в отдельном потоке,
    // чтобы не блокировать работу диктофона.
    let (transcribe_tx, transcribe_rx) = mpsc::channel::<TranscriptionJob>();
    let transcriber_state = state.clone();
    thread::spawn(move || run_transcriber(transcriber_state, transcribe_rx));

    // Оборачиваем весь цикл диктофона в catch_unwind: паника в любом месте
    // (запись, чтение настроек, работа с каналами) НЕ должна отравлять мьютекс
    // и «убивать» GUI.
    let state_for_log = state.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_recorder_inner(state, key_rx, transcribe_tx);
    }));
    if let Err(_panic) = result {
        append_log(
            &state_for_log,
            "Критическая ошибка в потоке диктофона. \
             Приложение продолжает работу, но запись может быть недоступна."
                .to_string(),
        );
    }
}

fn run_recorder_inner(
    state: Arc<Mutex<AppState>>,
    key_rx: Receiver<KeyCommand>,
    transcribe_tx: mpsc::Sender<TranscriptionJob>,
) {
    let initial = read_settings(&state);
    let mut current_mode = initial.press_mode;
    // Устройство, выбранное в настройках сейчас (для лога при смене) и устройство
    // открытого потока захвата.
    let mut current_device = initial.input_device.clone();
    let mut capture_device: Option<String> = None;
    let mut capture: Option<AudioCapture> = None;

    // Готовность: микрофон закрыт и приложение ждёт нажатия клавиши.
    append_log(
        &state,
        format!(
            "Готов к записи. Зажмите {} и говорите. Микрофон включается только на время записи.",
            hotkey_label(&initial)
        ),
    );

    let mut session = RecordingSession {
        buffer: Vec::new(),
        active: false,
        tail_start: None,
        smoothed_level: 0.0,
    };
    let tail_duration = Duration::from_millis(200);
    let mut devices_refreshed_at = Instant::now();
    let devices_refresh = Duration::from_secs(2);

    loop {
        // Свежие настройки применяются уже в этой итерации.
        let settings = read_settings(&state);

        // Сменилось устройство ввода — запоминаем новый выбор. Сам поток захвата
        // пересоздастся при следующем нажатии клавиши (`open_capture_if_needed`).
        if settings.input_device != current_device {
            current_device = settings.input_device.clone();
            append_log(
                &state,
                match &current_device {
                    Some(name) => format!("Устройство ввода: {name}"),
                    None => "Устройство ввода: по умолчанию".to_string(),
                },
            );
        }

        // Периодически обновляем список устройств ввода: новые
        // подключённые микрофоны появляются в списке без перезапуска.
        if devices_refreshed_at.elapsed() >= devices_refresh {
            devices_refreshed_at = Instant::now();
            let fresh = list_input_devices();
            if let Ok(mut s) = state.lock()
                && s.devices != fresh
            {
                s.devices = fresh;
                append_log(&state, "Список устройств ввода обновлён".to_string());
            }
        }

        // Сменился режим клавиши во время записи — аккуратно завершаем отрезок.
        if settings.press_mode != current_mode {
            if session.active {
                session.tail_start = Some(Instant::now());
            }
            current_mode = settings.press_mode;
        }

        // Сначала обрабатываем команды от горячей клавиши, если они есть.
        while let Ok(cmd) = key_rx.try_recv() {
            match (cmd, settings.press_mode) {
                (KeyCommand::Start, PressMode::PushToTalk) => {
                    if session.active {
                        // Клавишу снова зажали, пока дослушивался «хвост», —
                        // продолжаем ту же запись, а не начинаем новую (иначе
                        // сказанное до этого потерялось бы). Повторное событие
                        // нажатия при уже идущей записи просто игнорируем.
                        session.tail_start = None;
                    } else {
                        let _ = start_recording(
                            &mut capture,
                            &mut capture_device,
                            &settings,
                            &state,
                            &mut session,
                        );
                    }
                }
                (KeyCommand::Stop, PressMode::PushToTalk) => {
                    // Клавиша отпущена — начинаем отсчёт короткого «хвоста».
                    session.tail_start = Some(Instant::now());
                }
                (KeyCommand::Start, PressMode::Toggle) => {
                    if session.active {
                        // Повторное нажатие — останавливаем запись.
                        session.tail_start = Some(Instant::now());
                    } else {
                        let _ = start_recording(
                            &mut capture,
                            &mut capture_device,
                            &settings,
                            &state,
                            &mut session,
                        );
                    }
                }
                (KeyCommand::Stop, PressMode::Toggle) => {
                    // Отпускание клавиши в режиме «вкл/выкл» ничего не делает.
                }
            }
        }

        // Если идёт запись — забираем сэмплы из потока и копим в буфер.
        if session.active {
            if let Some(c) = &capture {
                let gain = settings.input_gain;
                let mut new_count = 0usize;
                while let Ok(sample) = c.rx.try_recv() {
                    let amplified = if (gain - 1.0).abs() > 0.01 {
                        ((sample as f32 * gain).clamp(i16::MIN as f32, i16::MAX as f32)) as i16
                    } else {
                        sample
                    };
                    session.buffer.push(amplified);
                    new_count += 1;
                }
                // Индикатор уровня: экспоненциальная средняя RMS новых сэмплов.
                if new_count > 0 {
                    let start = session.buffer.len() - new_count;
                    let count = new_count as f64;
                    let rms: f64 = (session.buffer[start..]
                        .iter()
                        .map(|&s| (s as f64 / i16::MAX as f64).powi(2))
                        .sum::<f64>()
                        / count)
                        .sqrt();
                    let rms_f32 = rms as f32;
                    session.smoothed_level = session.smoothed_level.mul_add(0.6, rms_f32 * 0.4);
                    set_level(&state, session.smoothed_level);
                }
            }

            // Дослушиваем небольшой «хвост» после останова записи,
            // чтобы не обрезать последние звуки.
            if let Some(start) = session.tail_start
                && start.elapsed() >= tail_duration
            {
                session.active = false;
                session.tail_start = None;
                set_recording(&state, false);
                session.smoothed_level = 0.0;
                set_level(&state, 0.0);

                // Забираем параметры потока и сразу закрываем микрофон —
                // до следующего нажатия приложение не слушает.
                let sample_rate = capture.as_ref().map(|c| c.sample_rate).unwrap_or(0);
                capture = None;
                capture_device = None;

                // Сохраняем аудио (если включено) и отправляем запись на
                // транскрибацию с текущими настройками.
                let samples = std::mem::take(&mut session.buffer);
                let keep_audio = settings.keep_audio
                    && match save_wav(&samples, sample_rate, OUTPUT_FILE) {
                        Ok(()) => {
                            append_log(
                                &state,
                                format!("Файл сохранён: {OUTPUT_FILE}. Начинаю расшифровку..."),
                            );
                            true
                        }
                        Err(err) => {
                            // Файл не записался — расшифровываем из памяти,
                            // чтобы сказанное не пропало.
                            append_log(
                                &state,
                                format!(
                                    "Не удалось сохранить {OUTPUT_FILE}: {err}. \
                                     Начинаю расшифровку из памяти..."
                                ),
                            );
                            false
                        }
                    };
                if !settings.keep_audio {
                    append_log(
                        &state,
                        "Хранение аудио отключено — расшифровываю запись из памяти...".to_string(),
                    );
                }
                let _ = transcribe_tx.send(TranscriptionJob {
                    wav_path: OUTPUT_FILE.to_string(),
                    samples,
                    sample_rate,
                    auto_punctuation: settings.auto_punctuation,
                    noise_reduction: settings.noise_reduction,
                    keep_audio,
                });
            }
        }

        // Во время записи опрашиваем чаще (сэмплы и индикатор уровня),
        // в ожидании — реже, чтобы не нагружать процессор впустую.
        thread::sleep(Duration::from_millis(if session.active { 2 } else { 10 }));
    }
}

/// Кэш загруженных движков распознавания.
///
/// Модель грузим лениво (при первой записи) и держим загруженной, чтобы не
/// платить за загрузку при каждом нажатии клавиши. В памяти держится только
/// выбранный в настройках движок — при переключении другой выгружается.
#[derive(Default)]
struct EngineCache {
    /// Загруженный контекст Whisper (None — ещё не грузили или устарел).
    whisper: Option<WhisperContext>,
    /// Путь модели Whisper, которая сейчас в `whisper`. Сменился — перезагружаем.
    whisper_model: Option<String>,
    /// Загруженная модель GigaAM (None — ещё не грузили или устарела).
    gigaam: Option<GigaAMModel>,
    /// Папка модели GigaAM, которая сейчас в `gigaam`. Сменилась — перезагружаем.
    gigaam_dir: Option<String>,
}

impl EngineCache {
    /// Сверяет загруженные модели с настройками: если выбранный движок или
    /// путь модели сменились — сбрасывает кэш (перезагрузка произойдёт
    /// при следующей записи).
    fn sync(&mut self, settings: &Settings) {
        match settings.speech_engine {
            SpeechEngine::Whisper => {
                let resolved = resolve_model_path(settings.model_path.as_deref());
                if resolved != self.whisper_model {
                    // Модель сменилась или пропала — перезагрузим при следующей записи.
                    self.whisper = None;
                    self.whisper_model = resolved;
                }
                // Другой движок не нужен — освобождаем память.
                self.gigaam = None;
                self.gigaam_dir = None;
            }
            SpeechEngine::GigaAM => {
                let resolved =
                    models::resolve_gigaam_dir().map(|dir| dir.to_string_lossy().into_owned());
                if resolved != self.gigaam_dir {
                    self.gigaam = None;
                    self.gigaam_dir = resolved;
                }
                self.whisper = None;
                self.whisper_model = None;
            }
        }
    }
}

/// Поток транскрибации: держит движок распознавания загруженным и расшифровывает записи.
///
/// Поток живёт всё время работы программы. Даже если отдельная запись
/// вызывает панику (например, не хватило памяти), поток переживает сбой и
/// продолжает принимать новые записи — приложение не «молчит».
fn run_transcriber(state: Arc<Mutex<AppState>>, rx: Receiver<TranscriptionJob>) {
    // Модель грузим лениво (при первой записи), чтобы не тормозить запуск программы.
    let mut cache = EngineCache::default();

    loop {
        // Ждём запись; раз в полсекунды сверяем кэш моделей с настройками,
        // чтобы при смене движка ненужная модель выгружалась из памяти.
        let next_job = match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(job) => Some(job),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        };

        // Свежие настройки: движок, путь к модели и язык распознавания.
        let settings = read_settings(&state);
        cache.sync(&settings);

        if let Some(job) = next_job {
            process_job(&state, &settings, &mut cache, job);
        }
    }
}

/// Расшифровывает одну запись, пишет итог в журналы и вставляет текст.
fn process_job(
    state: &Arc<Mutex<AppState>>,
    settings: &Settings,
    cache: &mut EngineCache,
    job: TranscriptionJob,
) {
    let TranscriptionJob {
        wav_path,
        samples,
        sample_rate,
        auto_punctuation,
        noise_reduction,
        keep_audio,
    } = job;

    // Оборачиваем работу в catch_unwind: поток не должен умирать.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        transcribe_and_save(
            &wav_path,
            &samples,
            sample_rate,
            auto_punctuation,
            noise_reduction,
            settings.model_path.as_deref(),
            &settings.language,
            settings.speech_engine,
            cache,
        )
    }));

    let audio_note = if keep_audio {
        format!("Запись: {wav_path}")
    } else {
        "Аудиофайл не сохранён (настройка «Сохранять аудио записи»)".to_string()
    };

    match result {
        Ok(Ok(outcome)) => {
            // Сохраняем реплику в журнал (память + диск) до вставки,
            // чтобы запись не потерялась, даже если вставка не удастся.
            record_replica(state, &outcome, settings.privacy_mode);
            if outcome.word_count == 0 {
                append_log(
                    state,
                    format!(
                        "Речь не распознана: запись слишком тихая, короткая или без речевой \
                         энергии (длина {:.2} с, уровень {:.3}). {audio_note}.",
                        outcome.audio_secs, outcome.signal_level
                    ),
                );
                return;
            }

            // В режиме приватности текст в журнал событий не пишем.
            let texts = if settings.privacy_mode {
                String::new()
            } else {
                format!(
                    "\n  Сырой текст: {}\n  После обработки: {}",
                    raw_display(&outcome.raw_text),
                    outcome.text
                )
            };
            append_log(
                state,
                format!(
                    "Расшифровка готова: {} (слов: {}). {audio_note} (длина {:.2} с, \
                     уровень {:.3}).{texts}",
                    outcome.txt_path, outcome.word_count, outcome.audio_secs, outcome.signal_level,
                ),
            );

            // Вставляем распознанный текст в приложение, где стоит курсор.
            #[cfg(target_os = "windows")]
            insert_or_copy(state, &outcome);
        }
        Ok(Err(err)) => append_log(state, format!("Ошибка расшифровки: {err}. {audio_note}")),
        Err(_panic) => {
            // Непредвиденный сбой (память и т.п.) — сообщаем и продолжаем.
            append_log(
                state,
                format!("Сбой при расшифровке. {audio_note}. Следующая запись попробует ещё раз."),
            );
            // Перезагружаем модель заново — исходная могла остаться в
            // недопустимом состоянии.
            *cache = EngineCache::default();
        }
    }
}

/// Вставляет текст под курсор (SendInput с KEYEVENTF_UNICODE — работает
/// с любым окном и не трогает буфер обмена). Если вставка не удалась, текст
/// не пропадает: он кладётся в буфер обмена как запасной путь.
#[cfg(target_os = "windows")]
fn insert_or_copy(state: &Arc<Mutex<AppState>>, outcome: &TranscribeOutcome) {
    if outcome.text.trim().is_empty() {
        return;
    }
    let message = match insert::insert_text_at_cursor(&outcome.text) {
        Ok(()) => "Текст вставлен в активное окно (под курсор).".to_string(),
        Err(err) => match insert::copy_to_clipboard(&outcome.text) {
            Ok(()) => format!(
                "Не удалось вставить текст в активное окно ({err}). \
                 Текст скопирован в буфер обмена — вставьте его сами (Ctrl+V)."
            ),
            Err(clip_err) => format!(
                "Не удалось вставить текст в окно ({err}) и скопировать \
                 в буфер обмена ({clip_err}). Текст сохранён: {}.",
                outcome.txt_path
            ),
        },
    };
    append_log(state, message);
}

/// Расшифровывает запись выбранной моделью и записывает текст в .txt рядом с WAV.
///
/// Возвращает результат с текстом, числом слов и диагностикой
/// (0 слов — если модель не услышала речи). Если пост-обработка не
/// справилась, пишется «сырой» текст, чтобы пользователь не
/// остался вообще без расшифровки.
///
/// Аудио готовится одинаково для обеих моделей (16 кГц моно f32), затем
/// расшифровка идёт через выбранный движок. `model_override` — путь к модели
/// Whisper из настроек (если задан), `language` — код языка или «auto».
#[allow(clippy::too_many_arguments)]
fn transcribe_and_save(
    wav_path: &str,
    samples: &[i16],
    sample_rate: u32,
    auto_punctuation: bool,
    noise_reduction: bool,
    model_override: Option<&str>,
    language: &str,
    engine: SpeechEngine,
    cache: &mut EngineCache,
) -> Result<TranscribeOutcome, Box<dyn std::error::Error>> {
    // Обе модели ждут моно-аудио f32 с частотой 16 кГц: делаем ресемплинг,
    // обрезаем тишину и нормализуем громкость для лучшего распознавания.
    let resampled = audio::resample_to_whisper(samples, sample_rate);
    let trimmed = audio::trim_silence(&resampled);
    let mut audio = trimmed.to_vec();
    // Шумоподавление — по желанию, до нормализации громкости.
    if noise_reduction {
        audio::apply_noise_reduction(&mut audio);
    }
    audio::normalize_audio(&mut audio);

    // Быстрое нажатие/отпускание клавиши даёт почти пустую запись: речи нет,
    // но модели иногда «галлюцинируют» («Продолжение следует»). Если в звуке
    // нет энергии — не вставляем ничего: пишем пустой txt и возвращаем 0 слов.
    if !audio::has_speech_energy(&audio) {
        return no_speech_outcome(wav_path, &audio);
    }

    // Очень короткая запись — это не слово, а щелчок/дребезг клавиши или
    // случайный звук. Модель на таком обрывке «додумывает» правдоподобную
    // фразу, поэтому отдаём ей только осмысленно длинный клип.
    const MIN_SPEECH_SECONDS: f64 = 0.25;
    if audio.len() as f64 / audio::TARGET_RATE < MIN_SPEECH_SECONDS {
        return no_speech_outcome(wav_path, &audio);
    }

    let raw = match engine {
        SpeechEngine::Whisper => {
            transcribe_whisper(model_override, language, &mut cache.whisper, &audio)?
        }
        SpeechEngine::GigaAM => transcribe_gigaam(&mut cache.gigaam, &audio)?,
    };

    // Пост-обработка: чистка от слов-паразитов, пунктуация, абзацы.
    // Если автоматическая пунктуация выключена, текст оставляем «как сказан».
    let processed = postprocess::postprocess_text(&raw, auto_punctuation);
    // Если пост-обработка «съела» текст, а в записи явно была речь — пишем
    // исходный вариант, чтобы не потерять информацию.
    let final_text = if !processed.trim().is_empty() {
        processed
    } else {
        raw.trim().to_string()
    };

    let word_count = final_text.split_whitespace().count();

    // Заменяем расширение .wav на .txt — txt окажется рядом с записью голоса.
    let txt_path = replace_wav_extension(wav_path, "txt");
    write_text_file(&txt_path, &final_text)?;

    Ok(TranscribeOutcome {
        txt_path,
        word_count,
        text: final_text,
        raw_text: raw,
        audio_secs: audio.len() as f64 / audio::TARGET_RATE,
        signal_level: peak_amplitude(&audio),
    })
}

/// Расшифровывает аудио через Whisper (GGML-модель). Контекст грузится при
/// первом использовании и переиспользуется дальше.
fn transcribe_whisper(
    model_override: Option<&str>,
    language: &str,
    context: &mut Option<WhisperContext>,
    audio: &[f32],
) -> Result<String, Box<dyn std::error::Error>> {
    let ctx = match context {
        Some(ctx) => ctx,
        None => {
            let model_path = match resolve_model_path(model_override) {
                Some(path) => path,
                None => {
                    let message =
                        "Файл модели не найден. Положите модель рядом с программой (см. README), \
                          укажите путь в настройках или переменной окружения WHISPER_MODEL"
                            .to_string();
                    return Err(message.into());
                }
            };
            // Flash-внимание: заметно ускоряет инференс на видеокарте и
            // экономит её память. На CPU-сборке просто игнорируется.
            let cparams = WhisperContextParameters {
                flash_attn: true,
                ..Default::default()
            };
            let ctx = WhisperContext::new_with_params(model_path, cparams)?;
            context.insert(ctx)
        }
    };

    let mut state = ctx.create_state()?;
    // Параметры декодинга общие с examples/transcribe_test.rs.
    let params = whisper_config::full_params(language);
    state.full(params, audio)?;

    // Собираем распознанный текст из всех сегментов.
    let mut raw = String::new();
    let num_segments = state.full_n_segments();
    for i in 0..num_segments {
        let segment = state.get_segment(i).ok_or("Нет сегмента результата")?;
        let text = segment.to_str_lossy()?;
        if !text.trim().is_empty() {
            raw.push_str(text.trim());
            raw.push(' ');
        }
    }
    Ok(raw)
}

/// Расшифровывает аудио через GigaAM v3 e2e-ctc (ONNX). Модель грузится при
/// первом использовании из папки, куда её скачал пользователь.
fn transcribe_gigaam(
    gigaam: &mut Option<GigaAMModel>,
    audio: &[f32],
) -> Result<String, Box<dyn std::error::Error>> {
    let model = match gigaam {
        Some(model) => model,
        None => {
            let dir = match models::resolve_gigaam_dir() {
                Some(dir) => dir,
                None => {
                    let message = "Модель GigaAM v3 не скачана. Откройте «Настройки → Модель \
                                   распознавания» и нажмите «Скачать модель GigaAM v3»."
                        .to_string();
                    return Err(message.into());
                }
            };
            // В папке лежат model.onnx и vocab.txt (см. models.rs).
            let model = GigaAMModel::load(&dir, &Quantization::default())?;
            gigaam.insert(model)
        }
    };

    // GigaAM e2e-ctc сам расставляет пунктуацию и понимает русскую речь,
    // поэтому язык в настройках на него не влияет.
    let result = model.transcribe(audio, &TranscribeOptions::default())?;
    Ok(result.text)
}

/// Пишет текст в файл и проверяет, что запись действительно прошла.
fn write_text_file(path: &str, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(path, text)?;
    let meta = std::fs::metadata(path)?;
    if meta.len() != text.len() as u64 {
        return Err(format!(
            "Файл {path} записался не полностью (ожидалось {} байт, получено {})",
            text.len(),
            meta.len()
        )
        .into());
    }
    Ok(())
}

/// Заменяет расширение пути на новое (например, output.wav -> output.txt).
fn replace_wav_extension(path: &str, new_ext: &str) -> String {
    let stem = std::path::Path::new(path).with_extension(new_ext);
    stem.to_string_lossy().into_owned()
}

/// Ищет файл модели Whisper в нескольких стандартных местах.
///
/// Порядок поиска: путь из настроек пользователя, переменная WHISPER_MODEL,
/// папка с исполняемым файлом, текущая папка, папка models/. Возвращает
/// первый найденный путь.
fn resolve_model_path(model_override: Option<&str>) -> Option<String> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // Путь из настроек имеет приоритет над остальными источниками.
    if let Some(configured) = model_override.filter(|path| !path.trim().is_empty()) {
        candidates.push(std::path::PathBuf::from(configured.trim()));
    }
    if let Ok(env_model) = std::env::var("WHISPER_MODEL") {
        candidates.push(std::path::PathBuf::from(env_model));
    }
    if let Ok(exe_path) = std::env::current_exe()
        && let Some(exe_dir) = exe_path.parent()
    {
        candidates.push(exe_dir.join(MODEL_FILE));
    }
    candidates.push(std::path::PathBuf::from(MODEL_FILE));
    let models_dir = std::path::PathBuf::from("models");
    candidates.push(models_dir.join(MODEL_FILE));

    candidates
        .into_iter()
        .find(|path| path.is_file())
        .map(|path| path.to_string_lossy().into_owned())
}

/// Обновляет флаг записи в общем состоянии.
fn set_recording(state: &Arc<Mutex<AppState>>, recording: bool) {
    if let Ok(mut s) = state.lock() {
        s.recording = recording;
    }
}

/// Обновляет текущий уровень входного сигнала (для индикатора).
fn set_level(state: &Arc<Mutex<AppState>>, level: f32) {
    if let Ok(mut s) = state.lock() {
        s.input_level = level;
    }
}

/// Записывает статусное сообщение в общее состояние.
fn set_status(state: &Arc<Mutex<AppState>>, status: String) {
    if let Ok(mut s) = state.lock() {
        s.status = status;
    }
}

/// Обновляет прогресс скачивания модели (для полосы в окне настроек).
fn set_download_progress(state: &Arc<Mutex<AppState>>, downloaded: u64, total: u64) {
    if let Ok(mut s) = state.lock() {
        s.download.downloaded = downloaded;
        s.download.total = total;
    }
}

/// Помечает, что скачивание модели остановлено (успех или ошибка).
fn set_download_active(state: &Arc<Mutex<AppState>>, active: bool) {
    if let Ok(mut s) = state.lock() {
        s.download.active = active;
    }
}

/// Запускает скачивание модели GigaAM v3 e2e-ctc в фоновом потоке.
///
/// Модель не входит в базовую поставку (в отличие от Whisper), её пользователь
/// скачивает один раз из окна настроек. Прогресс пишется в `AppState.download`,
/// по завершении — сообщение в журнал.
fn start_gigaam_download(state: Arc<Mutex<AppState>>) {
    if let Ok(mut s) = state.lock() {
        s.download = DownloadState {
            active: true,
            downloaded: 0,
            total: 0,
        };
    }

    thread::spawn(move || {
        let dest_dir = models::gigaam_download_dir();
        append_log(
            &state,
            format!(
                "Скачиваю модель GigaAM v3 (e2e-ctc) в {}...",
                dest_dir.display()
            ),
        );

        let model_dest = dest_dir.join(models::GIGAAM_MODEL_FILE);
        let vocab_dest = dest_dir.join(models::GIGAAM_VOCAB_FILE);
        let result = (|| -> Result<(), String> {
            download::download_file(models::GIGAAM_MODEL_URL, &model_dest, |done, total| {
                set_download_progress(&state, done, total);
            })?;
            // Словарь крошечный, отдельный прогресс для него не нужен.
            download::download_file(models::GIGAAM_VOCAB_URL, &vocab_dest, |_, _| {})?;
            Ok(())
        })();

        set_download_active(&state, false);
        match result {
            Ok(()) => append_log(
                &state,
                format!(
                    "Модель GigaAM v3 скачана: {}. Следующие записи расшифровываются через неё.",
                    dest_dir.display()
                ),
            ),
            Err(err) => append_log(
                &state,
                format!("Скачивание модели GigaAM не удалось: {err}"),
            ),
        }
    });
}

/// Дописывает строку с датой и временем в журнал `voiceai.log` (без статуса
/// в окне). Слишком большой журнал переименовывается в `voiceai.log.old`.
fn write_plain_log(message: &str) {
    use std::io::Write;

    if std::fs::metadata(LOG_FILE).is_ok_and(|meta| meta.len() > LOG_MAX_BYTES) {
        let _ = std::fs::rename(LOG_FILE, format!("{LOG_FILE}.old"));
    }
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
    {
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(file, "[{now}] {message}");
    }
}

/// Показывает модальный диалог Windows (например, «приложение уже запущено»).
#[cfg(target_os = "windows")]
fn show_message_box(title: &str, message: &str) {
    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONINFORMATION, MB_OK, MessageBoxW};

    let title_wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    let message_wide: Vec<u16> = message.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        MessageBoxW(
            HWND::default(),
            message_wide.as_ptr(),
            title_wide.as_ptr(),
            MB_OK | MB_ICONINFORMATION,
        );
    }
}

/// Показывает сообщение в окне программы и дописывает его в журнал `voiceai.log`.
///
/// Журнал — страховка на случай, когда окно неудобно читать: по нему всегда
/// видно, что программа делала и что пошло не так.
fn append_log(state: &Arc<Mutex<AppState>>, message: String) {
    write_plain_log(&message);
    set_status(state, message);
}

/// Выбрасывает все сэмплы, уже ожидающие в канале, не сохраняя их.
fn discard_pending(rx: &Receiver<i16>) {
    while rx.try_recv().is_ok() {}
}

/// Открытый поток захвата звука: сам поток, приёмник сэмплов и частота.
///
/// Пока структура жива, микрофон открыт; при удалении он закрывается.
struct AudioCapture {
    /// Приёмник моно-сэмплов (i16).
    rx: Receiver<i16>,
    /// Реальная частота дискретизации микрофона.
    sample_rate: u32,
    /// Сам поток захвата — держит микрофон открытым, пока жив.
    stream: cpal::Stream,
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        // Явно останавливаем поток (на Windows это WASAPI `audio_client.Stop()`).
        // Без этого некоторые драйверы/Windows держат микрофон «активным» (индикатор
        // прослушки не гаснет), даже когда `cpal::Stream` уже освобождён.
        let _ = self.stream.pause();
    }
}

/// Запускает поток записи с указанного устройства ввода (или по умолчанию).
///
/// Возвращает живой поток и параметры захвата. Сэмплы уже объединены
/// в моно, поэтому частота соответствует числу моно-семплов в секунду.
fn capture_stream(
    device_name: Option<&str>,
    state: Arc<Mutex<AppState>>,
) -> Result<AudioCapture, Box<dyn std::error::Error>> {
    let host = cpal::default_host();

    let device = match device_name {
        Some(name) => find_input_device(&host, name)
            .ok_or_else(|| format!("Устройство ввода '{name}' не найдено"))?,
        None => host
            .default_input_device()
            .ok_or("Не найден микрофон (устройство ввода)")?,
    };

    let config = device
        .default_input_config()
        .map_err(describe_default_config_error)?;

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;

    let (tx, rx) = mpsc::channel::<i16>();

    // Строим поток захвата. На выходе всегда 16-битные сэмплы (i16) для записи в WAV.
    let stream = match config.sample_format() {
        // 32-битный звук с плавающей точкой — самый распространённый на Windows.
        // Конвертируем f32 в i16 прямо в потоке захвата.
        SampleFormat::F32 => {
            build_f32_stream(&device, &config.into(), channels, tx, state.clone())?
        }
        // 16-битные целые — используются как есть.
        SampleFormat::I16 => {
            build_i16_stream(&device, &config.into(), channels, tx, state.clone())?
        }
        other => {
            return Err(format!("Неподдерживаемый формат сэмплов: {other:?}").into());
        }
    };

    stream.play()?;

    Ok(AudioCapture {
        rx,
        sample_rate,
        stream,
    })
}

/// Создаёт поток захвата для 32-битного звука с плавающей точкой (f32).
///
/// Микрофоны на Windows часто отдают сэмплы как f32 в диапазоне [-1.0, 1.0].
/// Здесь они конвертируются в 16-битные целые (i16) для записи в WAV.
///
/// `channels` — число каналов микрофона. Если их несколько (например, стерео),
/// каналы усредняются в один моно-сэмпл, чтобы не растягивать звук во времени.
fn build_f32_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: mpsc::Sender<i16>,
    state: Arc<Mutex<AppState>>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let err_fn = move |err| {
        // Устройство пропало/занято прямо во время записи.
        if matches!(err, cpal::StreamError::DeviceNotAvailable) {
            append_log(
                &state,
                "Микрофон недоступен: устройство отключено или занято другим приложением."
                    .to_string(),
            );
        } else {
            append_log(&state, format!("Ошибка потока записи: {err}"));
        }
    };

    let stream = device
        .build_input_stream(
            config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Группируем сэмплы по `channels` штук (один аудио-фрейм).
                for frame in data.chunks(channels) {
                    // Усредняем все каналы фрейма в один моно-сэмпл.
                    let average = frame.iter().sum::<f32>() / channels as f32;
                    // Масштабируем на максимум i16 и округляем до целого.
                    let sample_i16 = (average.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    let _ = tx.send(sample_i16);
                }
            },
            err_fn,
            None,
        )
        .map_err(describe_build_error)?;

    Ok(stream)
}

/// Создаёт поток захвата для 16-битного целочисленного звука.
///
/// `channels` — число каналов микрофона. При нескольких каналах (стерео)
/// они усредняются в один моно-сэмпл, чтобы не растягивать звук во времени.
fn build_i16_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    tx: mpsc::Sender<i16>,
    state: Arc<Mutex<AppState>>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let err_fn = move |err| {
        // Устройство пропало/занято прямо во время записи.
        if matches!(err, cpal::StreamError::DeviceNotAvailable) {
            append_log(
                &state,
                "Микрофон недоступен: устройство отключено или занято другим приложением."
                    .to_string(),
            );
        } else {
            append_log(&state, format!("Ошибка потока записи: {err}"));
        }
    };

    let stream = device
        .build_input_stream(
            config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                // Группируем сэмплы по `channels` штук (один аудио-фрейм).
                for frame in data.chunks(channels) {
                    // Усредняем каналы в один моно-сэмпл, суммируя в i32,
                    // чтобы избежать переполнения.
                    let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                    let average = (sum / channels as i32) as i16;
                    let _ = tx.send(average);
                }
            },
            err_fn,
            None,
        )
        .map_err(describe_build_error)?;

    Ok(stream)
}

/// Понятное сообщение для «внутренней» ошибки звуковой подсистемы.
/// Чаще всего это отказ в доступе к микрофону.
fn describe_backend_error(message: &str) -> String {
    let lower = message.to_lowercase();
    let denied = ["access", "denied", "permission", "запрещ", "доступ"]
        .iter()
        .any(|word| lower.contains(word));
    if denied {
        "Нет доступа к микрофону. Разрешите доступ: Настройки Windows → \
         Конфиденциальность → Микрофон, и разрешите использование микрофона \
         приложениям."
            .to_string()
    } else {
        format!("Не удалось открыть устройство ввода: {message}").replace('\n', " ")
    }
}

/// Превращает ошибку запроса формата устройства в понятное сообщение.
fn describe_default_config_error(err: cpal::DefaultStreamConfigError) -> String {
    use cpal::DefaultStreamConfigError as E;
    match err {
        E::DeviceNotAvailable => {
            "Устройство ввода недоступно (отключено или занято другим приложением). \
             Подключите его заново или выберите другое устройство."
                .to_string()
        }
        E::StreamTypeNotSupported => {
            "Устройство не поддерживает захват звука. Выберите другое устройство ввода.".to_string()
        }
        E::BackendSpecific { err } => describe_backend_error(&err.description),
    }
}

/// Превращает ошибку создания потока захвата в понятное сообщение.
fn describe_build_error(err: cpal::BuildStreamError) -> String {
    use cpal::BuildStreamError as E;
    match err {
        E::DeviceNotAvailable => {
            "Устройство ввода недоступно (отключено или занято другим приложением). \
             Закройте программу, использующую микрофон, или подключите устройство заново."
                .to_string()
        }
        E::StreamConfigNotSupported => {
            "Устройство не поддерживает формат записи. Попробуйте другое устройство ввода."
                .to_string()
        }
        E::InvalidArgument => {
            "Устройство не поддерживает захват звука. Попробуйте другое устройство ввода."
                .to_string()
        }
        E::BackendSpecific { err } => describe_backend_error(&err.description),
        other => format!("Не удалось открыть устройство ввода: {other:?}"),
    }
}

/// Сохраняет вектор 16-битных сэмплов в WAV-файл, моно, 16 бит на сэмпл.
fn save_wav(
    samples: &[i16],
    sample_rate: u32,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_runs_gui_without_flags() {
        assert!(matches!(parse_cli(&[]), CliAction::Run));
        assert!(matches!(parse_cli(&["some.wav".into()]), CliAction::Run));
        assert!(matches!(
            parse_cli(&["--model".into(), "m.bin".into()]),
            CliAction::Run
        ));
    }

    #[test]
    fn cli_version_flag_wins_over_help_order() {
        // Первый же «флаг действия» решает: версия важнее справки.
        assert!(matches!(
            parse_cli(&["--version".into()]),
            CliAction::PrintVersion
        ));
        assert!(matches!(parse_cli(&["-V".into()]), CliAction::PrintVersion));
        assert!(matches!(
            parse_cli(&["-h".into(), "--version".into()]),
            CliAction::PrintHelp
        ));
        assert!(matches!(
            parse_cli(&["--version".into(), "-h".into()]),
            CliAction::PrintVersion
        ));
    }

    #[test]
    fn cli_help_flag() {
        assert!(matches!(
            parse_cli(&["--help".into()]),
            CliAction::PrintHelp
        ));
        assert!(matches!(parse_cli(&["-h".into()]), CliAction::PrintHelp));
    }
}
