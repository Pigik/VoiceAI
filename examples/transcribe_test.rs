// Пример для проверки полного конвейера вне GUI: WAV -> Whisper -> текст.
//
// Использование:
//   cargo run --example transcribe_test -- путь.wav [язык] [модель] [raw] [greedy]
//
// Аргументы (все, кроме пути к WAV, необязательные):
//   язык   — ru | en | auto (по умолчанию ru)
//   модель — путь к GGML-модели (по умолчанию ggml-large-v3-turbo.bin)
//   raw    — пропустить предобработку аудио (без обрезки тишины/нормализации)
//   greedy — использовать greedy-декодинг (по умолчанию beam search)
use std::time::Instant;

use voiceai::{audio, postprocess};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const DEFAULT_MODEL: &str = "ggml-large-v3-turbo.bin";

/// Число потоков Whisper: с CUDA главное на видеокарте, CPU-часть — до 8 потоков.
fn whisper_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| (n.get() as i32).min(8))
        .unwrap_or(4)
}

fn main() {
    let wav_path = std::env::args().nth(1).expect("Укажите путь к WAV");
    let lang = std::env::args().nth(2).unwrap_or_else(|| "ru".to_string());
    let model_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let raw = std::env::args().nth(4).as_deref() == Some("raw");
    let greedy = std::env::args().nth(5).as_deref() == Some("greedy");

    let started = Instant::now();
    println!("Загрузка модели {model_path}...");

    let cparams = WhisperContextParameters {
        flash_attn: true,
        ..Default::default()
    };
    let ctx =
        WhisperContext::new_with_params(&model_path, cparams).expect("Не удалось загрузить модель");
    println!("Модель загружена за {:?}", started.elapsed());

    // Читаем WAV.
    let mut reader = hound::WavReader::open(&wav_path).expect("Не удалось открыть WAV");
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    println!(
        "WAV: {} Гц, {} каналов, {} бит",
        sample_rate, spec.channels, spec.bits_per_sample
    );

    let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();

    // Готовим аудио: моно 16 кГц f32 + (опционально) обрезка тишины и
    // нормализация громкости — ровно так же, как делает приложение.
    let mut audio = audio::resample_to_whisper(&samples, sample_rate);
    if !raw {
        let trimmed = audio::trim_silence(&audio);
        audio = trimmed.to_vec();
        audio::normalize_audio(&mut audio);
    }

    let mut state = ctx.create_state().expect("create_state");

    // Те же настройки декодинга, что и в приложении.
    let mut params = if greedy {
        FullParams::new(SamplingStrategy::Greedy { best_of: 5 })
    } else {
        FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        })
    };
    if lang == "auto" {
        params.set_detect_language(true);
    } else {
        params.set_language(Some(lang.as_str()));
    }
    params.set_translate(false);
    params.set_no_timestamps(true);
    params.set_n_threads(whisper_threads());
    params.set_temperature_inc(0.2);

    println!("Распознавание...");
    let transcribe_started = Instant::now();
    state.full(params, &audio).expect("Ошибка транскрибации");
    println!("Распознавание заняло {:?}", transcribe_started.elapsed());

    // Собираем «сырой» текст из сегментов.
    let mut raw_text = String::new();
    let n = state.full_n_segments();
    for i in 0..n {
        if let Some(seg) = state.get_segment(i) {
            let text = seg.to_str_lossy().unwrap_or_default();
            println!("Сегмент [{i}]: {text}");
            if !text.trim().is_empty() {
                raw_text.push_str(text.trim());
                raw_text.push(' ');
            }
        }
    }

    println!();
    println!("=== Итоговый текст (пост-обработка) ===");
    println!("{}", postprocess::postprocess_text(&raw_text, true));

    println!("Готово за {:?}", started.elapsed());
}
