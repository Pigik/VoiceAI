//! Проверка полного конвейера без GUI: WAV -> подготовка аудио -> модель -> текст.
//!
//! Использование:
//!   cargo run --release --example transcribe_test -- запись.wav [язык] [модель] [флаги]
//!
//! Позиционные аргументы (все, кроме пути к WAV, необязательные):
//!   язык   — ru | en | auto | … (по умолчанию ru)
//!   модель — путь к GGML-модели Whisper (по умолчанию ggml-large-v3-turbo.bin)
//!
//! Флаги (в любом месте после пути к WAV):
//!   gigaam — распознавать GigaAM v3 e2e-ctc вместо Whisper
//!   raw    — пропустить подготовку аудио (обрезку тишины и нормализацию)
//!
//! Параметры декодинга Whisper те же, что и в приложении
//! (`voiceai::whisper_config::full_params`).

use std::time::Instant;

use transcribe_rs::onnx::Quantization;
use transcribe_rs::onnx::gigaam::GigaAMModel;
use transcribe_rs::{SpeechModel, TranscribeOptions};
use voiceai::{audio, models, postprocess, whisper_config};
use whisper_rs::{WhisperContext, WhisperContextParameters};

const DEFAULT_MODEL: &str = "ggml-large-v3-turbo.bin";

/// Флаги командной строки, которые не являются позиционными аргументами.
const FLAGS: &[&str] = &["gigaam", "raw"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let use_gigaam = args.iter().any(|arg| arg == "gigaam");
    let raw = args.iter().any(|arg| arg == "raw");
    let positional: Vec<&str> = args
        .iter()
        .map(String::as_str)
        .filter(|arg| !FLAGS.contains(arg))
        .collect();

    let Some(&wav_path) = positional.first() else {
        eprintln!(
            "Использование: transcribe_test запись.wav [язык] [модель] [gigaam] [raw]\n\
             Подробности — в начале файла examples/transcribe_test.rs."
        );
        std::process::exit(2);
    };
    let lang = positional.get(1).copied().unwrap_or("ru");
    let model_path = positional.get(2).copied().unwrap_or(DEFAULT_MODEL);

    // Читаем WAV.
    let mut reader = hound::WavReader::open(wav_path).expect("Не удалось открыть WAV");
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    println!(
        "WAV: {} Гц, {} каналов, {} бит",
        sample_rate, spec.channels, spec.bits_per_sample
    );
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        eprintln!("Ожидается моно WAV 16 бит (как output.wav, который пишет приложение).");
        std::process::exit(2);
    }

    let samples: Vec<i16> = reader
        .samples::<i16>()
        .map(|s| s.expect("Ошибка чтения сэмпла WAV"))
        .collect();

    // Готовим аудио: моно 16 кГц f32 + (опционально) обрезка тишины и
    // нормализация громкости — так же, как делает приложение.
    let mut audio = audio::resample_to_whisper(&samples, sample_rate);
    if !raw {
        audio = audio::trim_silence(&audio).to_vec();
        audio::normalize_audio(&mut audio);
    }

    let started = Instant::now();
    let raw_text = if use_gigaam {
        let dir = models::resolve_gigaam_dir()
            .expect("Модель GigaAM не найдена. Скачайте её через окно настроек приложения.");
        println!("Загрузка модели GigaAM из {}...", dir.display());
        let mut model =
            GigaAMModel::load(&dir, &Quantization::default()).expect("Не удалось загрузить модель");
        println!("Модель загружена за {:?}", started.elapsed());

        println!("Распознавание...");
        let transcribe_started = Instant::now();
        let result = model
            .transcribe(&audio, &TranscribeOptions::default())
            .expect("Ошибка транскрибации");
        println!("Распознавание заняло {:?}", transcribe_started.elapsed());
        result.text
    } else {
        println!("Загрузка модели {model_path}...");
        let cparams = WhisperContextParameters {
            flash_attn: true,
            ..Default::default()
        };
        let ctx = WhisperContext::new_with_params(model_path, cparams)
            .expect("Не удалось загрузить модель");
        println!("Модель загружена за {:?}", started.elapsed());

        let mut state = ctx.create_state().expect("create_state");
        let params = whisper_config::full_params(lang);

        println!("Распознавание...");
        let transcribe_started = Instant::now();
        state.full(params, &audio).expect("Ошибка транскрибации");
        println!("Распознавание заняло {:?}", transcribe_started.elapsed());

        // Собираем «сырой» текст из сегментов.
        let mut text = String::new();
        for i in 0..state.full_n_segments() {
            if let Some(seg) = state.get_segment(i) {
                let seg_text = seg.to_str_lossy().unwrap_or_default();
                println!("Сегмент [{i}]: {seg_text}");
                if !seg_text.trim().is_empty() {
                    text.push_str(seg_text.trim());
                    text.push(' ');
                }
            }
        }
        text
    };

    println!();
    println!("=== Итоговый текст (пост-обработка) ===");
    println!("{}", postprocess::postprocess_text(&raw_text, true));

    println!("Готово за {:?}", started.elapsed());
}
