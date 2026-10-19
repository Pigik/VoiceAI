//! Параметры распознавания Whisper.
//!
//! Общие для приложения (`src/main.rs`) и проверочного примера
//! (`examples/transcribe_test.rs`), чтобы пример проверял ровно те же
//! настройки декодинга, с которыми работает программа.

use whisper_rs::{FullParams, SamplingStrategy};

/// Число потоков CPU для Whisper. Основные вычисления уходят на видеокарту
/// (CUDA/Metal), а CPU-часть разгоняем до 8 потоков — этого с запасом
/// хватает, не перегружая систему.
pub fn whisper_threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| (n.get() as i32).min(8))
        .unwrap_or(4)
}

/// Параметры декодинга, с которыми работает приложение.
///
/// `language` — код языка («ru», «en», …), «auto» или пустая строка —
/// автоопределение языка.
pub fn full_params(language: &str) -> FullParams<'_, '_> {
    // Greedy best-of устойчивее к коротким и невнятным записям: делает
    // несколько проходов и берёт лучший вариант. Beam search на коротких
    // клипах давал выдуманный текст.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    // Для автоопределения передаём язык «auto». Флаг `detect_language` для
    // этого не подходит: с ним whisper.cpp только определяет язык и выходит,
    // не распознавая речь (результат всегда пустой).
    let language = if language.is_empty() {
        "auto"
    } else {
        language
    };
    params.set_language(Some(language));
    params.set_translate(false);
    params.set_no_timestamps(true);
    params.set_n_threads(whisper_threads());
    // Без повторных проходов с повышенной температурой: на мутных коротких
    // записях они усиливают «галлюцинации».
    params.set_temperature_inc(-1.0);
    // Не выдаём служебные/шумовые токены в текст.
    params.set_suppress_nst(true);
    params
}
