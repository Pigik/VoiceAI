// На Windows скрываем консольное окно: приложение запускается только как GUI.
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use voiceai::{audio, autostart, insert, postprocess, settings, single_instance};

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use global_hotkey::hotkey::{Code, HotKey};
use global_hotkey::{GlobalHotKeyEvent, GlobalHotKeyManager, HotKeyState};
use voiceai::settings::{PressMode, Settings};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const OUTPUT_FILE: &str = "output.wav";
/// Файл журнала событий (рядом с записью). Туда пишутся все важные шаги
/// работы: сохранение, расшифровка, ошибки. Помогает понять, что пошло не так,
/// когда окно программы не показывает деталей.
const LOG_FILE: &str = "voiceai.log";
/// Имя файла модели Whisper (GGML .bin). Модель будет искаться рядом
/// с программой, в текущей папке, в папке models/ или по переменной
/// окружения WHISPER_MODEL (путь из настроек имеет приоритет).
const MODEL_FILE: &str = "ggml-large-v3-turbo.bin";
/// Запасная модель (меньше по размеру). Используется, если основная
/// модель не найдена ни в одном из мест поиска.
const FALLBACK_MODEL_FILE: &str = "ggml-large-v3-turbo-q5_0.bin";

/// Число параллельных потоков для Whisper. Основные вычисления уходят на
/// видеокарту (CUDA), а CPU-часть (токенизация, часть декодов) разгоняем
/// до 8 потоков — этого с запасом хватает, не перегружая систему.
fn whisper_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| (n.get() as i32).min(8))
        .unwrap_or(4)
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
}

/// GUI-приложение: белое окно с текстом по центру и окном настроек.
struct DictophoneApp {
    state: Arc<Mutex<AppState>>,
    /// Открыто ли окно настроек.
    show_settings: bool,
    /// Куда экспортировать профиль настроек (путь в поле ввода).
    export_path: String,
    /// Откуда импортировать профиль настроек (путь в поле ввода).
    import_path: String,
    /// Иконка в системном трее (Windows) и состояние, для которого она нарисована.
    #[cfg(target_os = "windows")]
    tray: Option<tray_icon::TrayIcon>,
    #[cfg(target_os = "windows")]
    tray_recording: Option<bool>,
    #[cfg(not(target_os = "windows"))]
    _no_tray: (),
}

impl DictophoneApp {
    /// `open_settings` — открыть окно настроек сразу (первый запуск или ошибка конфига).
    fn new(state: Arc<Mutex<AppState>>, open_settings: bool) -> Self {
        Self {
            state,
            show_settings: open_settings,
            export_path: default_profile_path(),
            import_path: String::new(),
            #[cfg(target_os = "windows")]
            tray: None,
            #[cfg(target_os = "windows")]
            tray_recording: None,
            #[cfg(not(target_os = "windows"))]
            _no_tray: (),
        }
    }
}

/// Путь к файлу профиля по умолчанию (рядом с записью output.wav).
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
                    if ui.button("⚙ Настройки").clicked() {
                        self.show_settings = true;
                    }
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
                    "Используйте F1 для записи".to_string()
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

                    // Индикатор уровня сигнала во время записи (B-индикатор уровня).
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
        if self.show_settings {
            egui::Window::new("Настройки")
                .open(&mut self.show_settings)
                .collapsible(false)
                .resizable(false)
                .min_width(420.0)
                .show(ui.ctx(), |ui| {
                    // Выбор устройства ввода.
                    let selected_label = new_settings
                        .input_device
                        .clone()
                        .unwrap_or_else(|| "По умолчанию (авто)".to_string());
                    egui::ComboBox::from_label("Устройство ввода")
                        .selected_text(selected_label)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    new_settings.input_device.is_none(),
                                    "По умолчанию (авто)",
                                )
                                .clicked()
                            {
                                new_settings.input_device = None;
                            }
                            for device in &devices {
                                let checked =
                                    new_settings.input_device.as_deref() == Some(device.as_str());
                                if ui.selectable_label(checked, device).clicked() {
                                    new_settings.input_device = Some(device.clone());
                                }
                            }
                        });

                    ui.add_space(10.0);
                    // Усиление входного сигнала (C-09).
                    ui.add(
                        egui::Slider::new(&mut new_settings.input_gain, 0.0..=4.0)
                            .text("Усиление входа")
                            .logarithmic(true),
                    );
                    ui.add_space(2.0);
                    ui.label("1.0 — без изменений, больше — громче");

                    ui.add_space(10.0);
                    // Шумоподавление (C-16).
                    ui.checkbox(
                        &mut new_settings.noise_reduction,
                        "Подавление фонового шума",
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
                    // Язык распознавания (B-09).
                    let language_label = settings::language_label(&new_settings.language);
                    egui::ComboBox::from_label("Язык распознавания")
                        .selected_text(language_label)
                        .show_ui(ui, |ui| {
                            for (code, label) in settings::LANGUAGE_CHOICES {
                                if ui
                                    .selectable_label(
                                        new_settings.language == *code,
                                        *label,
                                    )
                                    .clicked()
                                {
                                    new_settings.language = code.to_string();
                                }
                            }
                        });
                    ui.add_space(2.0);
                    ui.label("«Автоопределение» — Whisper сам распознает язык");

                    ui.add_space(10.0);
                    // Путь к модели Whisper (B-08), пусто — автопоиск.
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

                    ui.add_space(10.0);
                    // Автозапуск при входе в систему (B-10).
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
                    ui.label("Выключает добавление точек и абзацев в распознанный текст");

                    ui.add_space(10.0);
                    ui.separator();
                    // Импорт/экспорт профиля настроек (B-15).
                    ui.horizontal(|ui| {
                        ui.label("Экспорт профиля:");
                        ui.text_edit_singleline(&mut self.export_path);
                        if ui.button("Экспорт").clicked() {
                            match settings::export_settings(&self.export_path, &new_settings) {
                                Ok(()) => append_log(
                                    &self.state,
                                    format!("Профиль настроек сохранён: {}", self.export_path),
                                ),
                                Err(err) => append_log(
                                    &self.state,
                                    format!("Экспорт профиля: {err}"),
                                ),
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
                                    new_settings = imported;
                                    append_log(
                                        &self.state,
                                        format!("Профиль настроек импортирован: {}", self.import_path),
                                    );
                                }
                                Err(err) => append_log(
                                    &self.state,
                                    format!("Импорт профиля: {err}"),
                                ),
                            }
                        }
                        if ui.button("Оставить как есть").clicked() {
                            self.import_path.clear();
                        }
                    });

                    ui.add_space(10.0);
                    ui.separator();
                    // Сброс настроек к значениям по умолчанию (B-07).
                    if ui
                        .button("Сбросить все настройки к значениям по умолчанию")
                        .clicked()
                    {
                        new_settings = Settings::default();
                    }
                });

            // Применяем изменения настроек, синхронизируем автозапуск и сохраняем.
            if new_settings != settings {
                if new_settings.auto_start != settings.auto_start {
                    match autostart::apply_autostart(new_settings.auto_start) {
                        Ok(message) => append_log(&self.state, message),
                        Err(err) => append_log(&self.state, format!("Автозапуск: {err}")),
                    }
                }
                if let Ok(mut state) = self.state.lock() {
                    state.settings = new_settings.clone();
                }
                match settings::save_settings(&new_settings) {
                    Ok(()) => {}
                    Err(err) => append_log(&self.state, format!("Сохранение настроек: {err}")),
                }
            }
        }
    }
}

/// Команды, которыми поток горячих клавиш общается с диктофоном.
enum KeyCommand {
    /// Клавиша записи нажата — начинаем копить звук.
    Start,
    /// Клавиша записи отпущена — сохраняем накопленный звук в файл.
    Stop,
}

/// Задание на транскрибацию: готовый WAV-файл и его сэмплы.
enum TranscriptionJob {
    Text {
        /// Путь к сохранённому WAV (рядом с ним создастся .txt).
        wav_path: String,
        /// Накопленные моно-сэмплы записи (родная частота микрофона).
        samples: Vec<i16>,
        /// Частота дискретизации, соответствующая сэмплам.
        sample_rate: u32,
        /// Применять ли автоматическую пунктуацию к распознанному тексту.
        auto_punctuation: bool,
        /// Применять ли подавление фонового шума при подготовке аудио.
        noise_reduction: bool,
    },
}

fn main() -> eframe::Result {
    // Одна копия приложения (B-13): вторая копия не должна мешать первой.
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

    // Загружаем настройки и фиксируем обстоятельства: первый запуск (B-01/B-03)
    // либо понятная ошибка в файле настроек (B-06).
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

    // Первый запуск: файл настроек создаётся прямо сейчас (B-03).
    if loaded.first_launch && settings::save_settings(&settings).is_err() {
        initial_status = "Не удалось создать файл настроек".to_string();
    }

    // Синхронизируем автозапуск системы с настройкой (B-10): если пользователь
    // включил его в конфиге, а в системе записи нет — добавляем её.
    if settings.auto_start
        && let Err(err) = autostart::apply_autostart(true)
    {
        initial_status = format!("Автозапуск: {err}");
    }

    let app_state = AppState {
        devices,
        settings,
        status: initial_status,
        ..Default::default()
    };
    let state = Arc::new(Mutex::new(app_state));

    // Канал команд от горячей клавиши к диктофону.
    let (key_tx, key_rx) = mpsc::channel::<KeyCommand>();

    // Регистрируем глобальный хоткей F1 (работает независимо от активного окна).
    let _hotkey_manager = register_f1_hotkey(state.clone(), key_tx);

    // Запускаем диктофон в отдельном потоке, чтобы GUI оставался отзывчивым.
    let recorder_state = state.clone();
    thread::spawn(move || run_recorder(recorder_state, key_rx));

    // Первый запуск или битый конфиг — открываем окно настроек сразу (B-01).
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
        Box::new(move |_cc| {
            Ok(Box::new(DictophoneApp::new(
                app_state_for_frame,
                open_settings,
            )))
        }),
    )
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

/// Регистрирует глобальную горячую клавишу F1 и запускает поток,
/// который превращает события нажатия/отпускания в команды диктофона.
fn register_f1_hotkey(
    state: Arc<Mutex<AppState>>,
    key_tx: mpsc::Sender<KeyCommand>,
) -> Option<GlobalHotKeyManager> {
    let manager = match GlobalHotKeyManager::new() {
        Ok(manager) => manager,
        Err(err) => {
            set_status(&state, format!("Ошибка горячих клавиш: {err}"));
            return None;
        }
    };

    let hotkey = HotKey::new(None, Code::F1);
    if let Err(err) = manager.register(hotkey) {
        set_status(&state, format!("Не удалось зарегистрировать F1: {err}"));
        return None;
    }

    // Поток, следящий за событиями глобального хоткея.
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

/// Основной цикл диктофона (режим удержания F1 или «вкл/выкл»).
///
/// Живёт в отдельном потоке и только обновляет `AppState`, а GUI его считывает.
/// Настройки (устройство ввода, режим клавиши, пунктуация) читает из общего
/// состояния и применяет без перезапуска всей программы.
fn run_recorder(state: Arc<Mutex<AppState>>, key_rx: Receiver<KeyCommand>) {
    // Канал заданий на транскрибацию. Транскрибатор живёт в отдельном потоке,
    // чтобы не блокировать работу диктофона.
    let (transcribe_tx, transcribe_rx) = mpsc::channel::<TranscriptionJob>();
    let transcriber_state = state.clone();
    thread::spawn(move || run_transcriber(transcriber_state, transcribe_rx));

    // Запускаем захват звука с выбранного (или устройства по умолчанию).
    let initial = read_settings(&state);
    let mut current_mode = initial.press_mode;

    let mut capture = match capture_stream(initial.input_device.as_deref(), state.clone()) {
        Ok(capture) => capture,
        Err(err) => {
            set_status(&state, format!("Ошибка записи: {err}"));
            return;
        }
    };
    let mut current_device = initial.input_device;

    // Готовность (B-02): микрофон открыт и приложение ждёт нажатия F1.
    append_log(&state, "Готов к записи. Зажмите F1 и говорите.".to_string());

    let mut buffer: Vec<i16> = Vec::new();
    let mut recording = false;
    let mut tail_start: Option<Instant> = None;
    let tail_duration = Duration::from_millis(200);
    let mut devices_refreshed_at = Instant::now();
    let devices_refresh = Duration::from_secs(2);
    let mut smoothed_level: f32 = 0.0;

    loop {
        // Свежие настройки применяются уже в этой итерации.
        let settings = read_settings(&state);

        // Сменилось устройство ввода — пересоздаём поток захвата.
        if settings.input_device != current_device {
            match capture_stream(settings.input_device.as_deref(), state.clone()) {
                Ok(new_capture) => {
                    // В новом канале уже могут ждать холостые сэмплы — выбрасываем их.
                    discard_pending(&new_capture.rx);
                    capture = new_capture;
                    current_device = settings.input_device.clone();
                    append_log(
                        &state,
                        match &current_device {
                            Some(name) => format!("Устройство ввода: {name}"),
                            None => "Устройство ввода: по умолчанию".to_string(),
                        },
                    );
                }
                Err(err) => {
                    // Устройство не открылось — оставляем старое и сообщаем (C-12, C-13).
                    append_log(&state, format!("Ошибка смены устройства: {err}"));
                    set_status(&state, format!("Не удалось сменить устройство: {err}"));
                }
            }
        }

        // Периодически обновляем список устройств ввода (C-05): новые
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
            if recording {
                tail_start = Some(Instant::now());
            }
            current_mode = settings.press_mode;
        }

        // Сначала обрабатываем команды от горячей клавиши, если они есть.
        while let Ok(cmd) = key_rx.try_recv() {
            match (cmd, settings.press_mode) {
                (KeyCommand::Start, PressMode::PushToTalk) => {
                    // Отбрасываем «холостые» сэмплы, накопленные до нажатия клавиши.
                    discard_pending(&capture.rx);
                    buffer.clear();
                    recording = true;
                    tail_start = None;
                    smoothed_level = 0.0;
                    set_level(&state, 0.0);
                    set_recording(&state, true);
                }
                (KeyCommand::Stop, PressMode::PushToTalk) => {
                    // Клавиша отпущена — начинаем отсчёт короткого «хвоста».
                    tail_start = Some(Instant::now());
                }
                (KeyCommand::Start, PressMode::Toggle) => {
                    if recording {
                        // Повторное нажатие — останавливаем запись.
                        tail_start = Some(Instant::now());
                    } else {
                        discard_pending(&capture.rx);
                        buffer.clear();
                        recording = true;
                        tail_start = None;
                        smoothed_level = 0.0;
                        set_level(&state, 0.0);
                        set_recording(&state, true);
                    }
                }
                (KeyCommand::Stop, PressMode::Toggle) => {
                    // Отпускание клавиши в режиме «вкл/выкл» ничего не делает.
                }
            }
        }

        // Если идёт запись — забираем сэмплы из потока и копим в буфер.
        if recording {
            let gain = settings.input_gain;
            let mut new_count = 0usize;
            while let Ok(sample) = capture.rx.try_recv() {
                let amplified = if (gain - 1.0).abs() > 0.01 {
                    ((sample as f32 * gain).clamp(i16::MIN as f32, i16::MAX as f32)) as i16
                } else {
                    sample
                };
                buffer.push(amplified);
                new_count += 1;
            }
            // Индикатор уровня (C-07): экспоненциальная средняя RMS новых сэмплов.
            if new_count > 0 {
                let start = buffer.len() - new_count;
                let count = new_count as f64;
                let rms: f64 = (buffer[start..]
                    .iter()
                    .map(|&s| (s as f64 / i16::MAX as f64).powi(2))
                    .sum::<f64>()
                    / count)
                    .sqrt();
                let rms_f32 = rms as f32;
                smoothed_level = smoothed_level.mul_add(0.6, rms_f32 * 0.4);
                set_level(&state, smoothed_level);
            }

            // Дослушиваем небольшой «хвост» после останова записи,
            // чтобы не обрезать последние звуки.
            if let Some(start) = tail_start
                && start.elapsed() >= tail_duration
            {
                recording = false;
                tail_start = None;
                set_recording(&state, false);
                smoothed_level = 0.0;
                set_level(&state, 0.0);
                match save_wav(&buffer, capture.sample_rate, OUTPUT_FILE) {
                    Ok(path) => {
                        append_log(
                            &state,
                            format!("Файл сохранён: {path}. Начинаю расшифровку..."),
                        );
                        // Отправляем запись на транскрибацию с текущими настройками
                        // пунктуации и шумоподавления.
                        let _ = transcribe_tx.send(TranscriptionJob::Text {
                            wav_path: path,
                            samples: buffer.clone(),
                            sample_rate: capture.sample_rate,
                            auto_punctuation: settings.auto_punctuation,
                            noise_reduction: settings.noise_reduction,
                        });
                    }
                    Err(err) => append_log(&state, format!("Ошибка сохранения: {err}")),
                }
            }
        } else {
            // Если запись не идёт — постоянно выбрасываем холостые сэмплы.
            discard_pending(&capture.rx);
        }

        thread::sleep(Duration::from_millis(1));
    }
}

/// Поток транскрибации: загружает модель Whisper один раз и расшифровывает записи.
///
/// Поток живёт всё время работы программы. Даже если отдельная запись
/// вызывает панику (например, не хватило памяти), поток переживает сбой и
/// продолжает принимать новые записи — приложение не «молчит».
fn run_transcriber(state: Arc<Mutex<AppState>>, rx: Receiver<TranscriptionJob>) {
    // Модель грузим лениво (при первой записи), чтобы не тормозить запуск программы.
    let mut context: Option<WhisperContext> = None;
    // Какую модель сейчас загрузили (путь). Сменился путь — перезагружаем.
    let mut loaded_model: Option<String> = None;

    loop {
        // Свежие настройки: путь к модели (B-08) и язык распознавания (B-09).
        let settings = read_settings(&state);
        let resolved_model = resolve_model_path(settings.model_path.as_deref());
        if resolved_model != loaded_model {
            // Модель сменилась или больше не находится — перезагружаем при следующей записи.
            context = None;
            loaded_model = resolved_model;
        }

        while let Ok(job) = rx.try_recv() {
            let TranscriptionJob::Text {
                wav_path,
                samples,
                sample_rate,
                auto_punctuation,
                noise_reduction,
            } = job;

            // Оборачиваем работу в catch_unwind: поток не должен умирать.
            let captured = settings.model_path.clone();
            let language = settings.language.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                transcribe_and_save(
                    &wav_path,
                    &samples,
                    sample_rate,
                    auto_punctuation,
                    noise_reduction,
                    captured.as_deref(),
                    &language,
                    &mut context,
                )
            }));

            match result {
                Ok(Ok((txt_path, 0, _))) => {
                    // Запись была, но речи в ней не распознано.
                    append_log(
                        &state,
                        format!(
                            "Записано {wav_path}, но речь не распознана: запись слишком тихая или короткая. Текст сохранён в {txt_path}."
                        ),
                    );
                }
                Ok(Ok((txt_path, words, text))) => {
                    append_log(
                        &state,
                        format!(
                            "Расшифровка готова: {txt_path} (слов: {words}). Запись: {wav_path}"
                        ),
                    );
                    // Вставляем распознанный текст в приложение, где стоит курсор
                    // (Win32: буфер обмена + Ctrl+V, поэтому работает с любым окном).
                    #[cfg(target_os = "windows")]
                    if !text.trim().is_empty() && words > 0 {
                        match insert::insert_text_at_cursor(&text) {
                            Ok(_) => append_log(
                                &state,
                                "Текст вставлен в активное окно (под курсор).".to_string(),
                            ),
                            Err(err) => append_log(
                                &state,
                                format!("Не удалось вставить текст в окно: {err}"),
                            ),
                        }
                    }
                }
                Ok(Err(err)) => append_log(
                    &state,
                    format!("Ошибка расшифровки: {err}. Запись сохранена: {wav_path}"),
                ),
                Err(_panic) => {
                    // Непредвиденный сбой (память и т.п.) — сообщаем и продолжаем.
                    append_log(
                        &state,
                        format!(
                            "Сбой при расшифровке записи {wav_path}. Следующая запись попробует ещё раз."
                        ),
                    );
                    // Перезагружаем модель заново — исходная могла остаться в
                    // недопустимом состоянии.
                    context = None;
                }
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
}

/// Расшифровывает запись через Whisper и записывает текст в .txt рядом с WAV.
///
/// Возвращает путь к текстовому файлу, число слов и сам текст
/// (0 слов — если Whisper не услышал речи). Если пост-обработка не
/// справилась, пишется «сырой» текст Whisper, чтобы пользователь не
/// остался вообще без расшифровки.
///
/// `model_override` — путь к модели из настроек (если задан), `language` —
/// код языка или «auto» для автоопределения.
#[allow(clippy::too_many_arguments)]
fn transcribe_and_save(
    wav_path: &str,
    samples: &[i16],
    sample_rate: u32,
    auto_punctuation: bool,
    noise_reduction: bool,
    model_override: Option<&str>,
    language: &str,
    context: &mut Option<WhisperContext>,
) -> Result<(String, usize, String), Box<dyn std::error::Error>> {
    // Загружаем модель при первом использовании и переиспользуем её дальше.
    let ctx = match context {
        Some(ctx) => ctx,
        None => {
            let model_path = resolve_model_path(model_override);
            let model_path = match model_path {
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

    // Whisper нужен аудио форматом f32 моно с частотой 16 кГц — делаем ресемплинг,
    // обрезаем тишину и нормализуем громкость для лучшего распознавания.
    let resampled = audio::resample_to_whisper(samples, sample_rate);
    let trimmed = audio::trim_silence(&resampled);
    let mut audio = trimmed.to_vec();
    // Шумоподавление (C-16) — по желанию, до нормализации громкости.
    if noise_reduction {
        audio::apply_noise_reduction(&mut audio);
    }
    audio::normalize_audio(&mut audio);

    let mut state = ctx.create_state()?;

    // Greedy best-of более устойчив к коротким и невнятным записям:
    // делает несколько проходов и берёт лучший вариант. Beam search при
    // этом давал на коротких клипах выдуманный текст.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    if language.is_empty() || language == "auto" {
        params.set_detect_language(true);
    } else {
        params.set_language(Some(language));
    }
    params.set_translate(false);
    params.set_no_timestamps(true);
    params.set_n_threads(whisper_threads());
    // Отключаем повторные проходы с повышенной температурой: на мутных
    // коротких записях они усиливают «галлюцинации», а без них результат
    // стабильнее и предсказуемее.
    params.set_temperature_inc(-1.0);
    // Не выдаём служебные/шумовые токены в текст — меньше мусора.
    params.set_suppress_nst(true);

    state.full(params, &audio)?;

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

    // Пост-обработка: чистка от слов-паразитов, пунктуация, абзацы.
    // Если автоматическая пунктуация выключена, текст оставляем «как сказан».
    let processed = postprocess::postprocess_text(&raw, auto_punctuation);
    // Если пост-обработка «съела» текст, а в записи явно была речь — пишем
    // исходный вариант Whisper, чтобы не потерять информацию.
    let final_text = if !processed.trim().is_empty() {
        processed
    } else {
        raw.trim().to_string()
    };

    let word_count = final_text.split_whitespace().count();

    // Заменяем расширение .wav на .txt — txt окажется рядом с записью голоса.
    let txt_path = replace_wav_extension(wav_path, "txt");
    write_text_file(&txt_path, &final_text)?;

    Ok((txt_path, word_count, final_text))
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
/// папка с исполняемым файлом, текущая папка, папка models/. Для каждой папки
/// сначала пробуется основной файл модели (обычно более качественная точная
/// версия), затем запасная. Возвращает первый найденный путь.
fn resolve_model_path(model_override: Option<&str>) -> Option<String> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();

    // Путь из настроек имеет приоритет над остальными источниками (B-08).
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
        candidates.push(exe_dir.join(FALLBACK_MODEL_FILE));
    }
    candidates.push(std::path::PathBuf::from(MODEL_FILE));
    candidates.push(std::path::PathBuf::from(FALLBACK_MODEL_FILE));
    let models_dir = std::path::PathBuf::from("models");
    candidates.push(models_dir.join(MODEL_FILE));
    candidates.push(models_dir.join(FALLBACK_MODEL_FILE));

    candidates
        .into_iter()
        .find(|path| path.exists() && path.is_file())
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

/// Пишет строку в журнал `voiceai.log` (без статуса в окне).
fn write_plain_log(message: &str) {
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
    {
        use std::io::Write;
        let _ = writeln!(file, "{message}");
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

/// Показывает сообщение в окне программы И дописывает его в журнал `voiceai.log`.
///
/// Журнал — страховка на случай, когда окно неудобно читать: по нему всегда
/// видно, что программа делала и что пошло не так. К каждой строке добавляется
/// время от запуска программы (в секундах).
fn append_log(state: &Arc<Mutex<AppState>>, message: String) {
    use std::sync::OnceLock;
    static PROGRAM_START: OnceLock<Instant> = OnceLock::new();

    let elapsed = PROGRAM_START.get_or_init(Instant::now).elapsed().as_secs();
    set_status(state, message.clone());

    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(LOG_FILE)
    {
        use std::io::Write;
        let _ = writeln!(file, "[+{elapsed} c] {message}");
    }
}

/// Выбрасывает все сэмплы, уже ожидающие в канале, не сохраняя их.
fn discard_pending(rx: &Receiver<i16>) {
    while rx.try_recv().is_ok() {}
}

/// Живой поток захвата звука: сам поток, приёмник сэмплов и его параметры.
///
/// Пока структура жива, запись идёт: поэтому её хранят в переменной, а не
/// «забывают» (раньше поток захвата `std::mem::forget`-ался, но это не давало
/// менять устройство ввода на лету).
struct CapturedCapture {
    /// Приёмник моно-сэмплов (i16).
    rx: Receiver<i16>,
    /// Реальная частота дискретизации микрофона.
    sample_rate: u32,
    /// Число каналов микрофона (пока не используется, оставлено для будущих задач).
    _channels: usize,
    /// Сам поток захвата — держит микрофон открытым, пока жив.
    _stream: cpal::Stream,
}

/// Запускает поток записи с указанного устройства ввода (или по умолчанию).
///
/// Возвращает живой поток и параметры захвата. Сэмплы уже объединены
/// в моно, поэтому частота соответствует числу моно-семплов в секунду.
fn capture_stream(
    device_name: Option<&str>,
    state: Arc<Mutex<AppState>>,
) -> Result<CapturedCapture, Box<dyn std::error::Error>> {
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
        SampleFormat::F32 => build_f32_stream(&device, &config.into(), channels, tx, state.clone())?,
        // 16-битные целые — используются как есть.
        SampleFormat::I16 => build_i16_stream(&device, &config.into(), channels, tx, state.clone())?,
        other => {
            return Err(format!("Неподдерживаемый формат сэмплов: {other:?}").into());
        }
    };

    stream.play()?;

    Ok(CapturedCapture {
        rx,
        sample_rate,
        _channels: channels,
        _stream: stream,
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
        // Устройство пропало/занято прямо во время записи (C-04, C-12).
        if matches!(err, cpal::StreamError::DeviceNotAvailable) {
            append_log(
                &state,
                "Микрофон недоступен: устройство отключено или занято другим приложением.".to_string(),
            );
        } else {
            append_log(&state, format!("Ошибка потока записи: {err}"));
        }
    };

    let stream = device.build_input_stream(
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
        // Устройство пропало/занято прямо во время записи (C-04, C-12).
        if matches!(err, cpal::StreamError::DeviceNotAvailable) {
            append_log(
                &state,
                "Микрофон недоступен: устройство отключено или занято другим приложением.".to_string(),
            );
        } else {
            append_log(&state, format!("Ошибка потока записи: {err}"));
        }
    };

    let stream = device.build_input_stream(
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

/// Превращает ошибку запроса формата устройства в понятное сообщение (C-12, C-13).
fn describe_default_config_error(err: cpal::DefaultStreamConfigError) -> String {
    use cpal::DefaultStreamConfigError as E;
    match err {
        E::DeviceNotAvailable => {
            "Устройство ввода недоступно (отключено или занято другим приложением). \
             Подключите его заново или выберите другое устройство."
                .to_string()
        }
        E::StreamTypeNotSupported => {
            "Устройство не поддерживает захват звука. Выберите другое устройство ввода."
                .to_string()
        }
        E::BackendSpecific { err } => {
            let message = err.description;
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
    }
}

/// Превращает ошибку создания потока захвата в понятное сообщение (C-12, C-13).
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
        E::BackendSpecific { err } => {
            // Чаще всего здесь — отказ в доступе к микрофону (C-13).
            let message = err.description;
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
        other => format!("Не удалось открыть устройство ввода: {other:?}"),
    }
}

/// Сохраняет вектор 16-битных сэмплов в WAV-файл, моно, 16 бит на сэмпл.
fn save_wav(
    samples: &[i16],
    sample_rate: u32,
    path: &str,
) -> Result<String, Box<dyn std::error::Error>> {
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

    Ok(path.to_string())
}
