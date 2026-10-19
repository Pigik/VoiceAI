//! Подготовка аудио для Whisper.
//!
//! Микрофон отдаёт моно- или стерео-поток i16 на своей «родной» частоте
//! (обычно 48 кГц, 44.1 кГц или 16 кГц). Whisper ожидает моно-сэмплы f32
//! с частотой 16 кГц и уровнем в диапазоне [-1.0, 1.0]. Этот модуль делает
//! конвертацию, убирает тишину и выравнивает громкость.

/// Частота, на которую Whisper ожидает моно-аудио: 16 кГц.
pub const TARGET_RATE: f64 = 16_000.0;

/// Полуширина sinc-ядра фильтра-пересэмплера (в сэмплах исходной частоты).
/// Больше отводов — точнее срез спектра, но медленнее. 16 отводов — хороший
/// баланс для речи.
const SINC_TAPS: f64 = 16.0;

/// Параметр остроты окна Кайзера. Чем больше, тем сильнее подавляются
/// боковые лепестки фильтра (меньше «звона» и алиасинга на краях полосы).
const KAISER_BETA: f64 = 7.0;

/// Длина окна анализа: 20 мс при 16 кГц.
const WINDOW: usize = 320;

/// Ниже этого RMS (по окну) запись считается тишиной.
const SILENCE_RMS: f32 = 0.002;

fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / 32768.0
}

/// RMS каждого полного окна `WINDOW` (хвост короче окна не учитывается).
fn window_rms(audio: &[f32]) -> Vec<f32> {
    audio
        .chunks_exact(WINDOW)
        .map(|seg| (seg.iter().map(|x| x * x).sum::<f32>() / WINDOW as f32).sqrt())
        .collect()
}

/// Нормированный sinc (ядро идеального низкочастотного фильтра).
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Окно Кайзера, обрезающее бесконечный sinc до конечной длины.
/// Приближение модифицированной функции Бесселя I0 рядом Тейлора.
fn kaiser_window(x: f64) -> f64 {
    let t = x / SINC_TAPS;
    if t.abs() >= 1.0 {
        return 0.0;
    }
    let arg = (1.0 - t * t).max(0.0).sqrt();

    let i0 = |z: f64| {
        let z2 = z * z;
        1.0 + z2 / 4.0
            + z2 * z2 / 64.0
            + z2 * z2 * z2 / 2304.0
            + z2 * z2 * z2 * z2 / 147456.0
            + z2 * z2 * z2 * z2 * z2 / 14745600.0
    };
    i0(KAISER_BETA * arg) / i0(KAISER_BETA)
}

/// Пересэмплирует моно i16-сэмплы с произвольной частоты до 16 кГц и
/// конвертирует их в f32, применяя оконный sinc-фильтр.
///
/// В отличие от простой линейной интерполяции такой фильтр честно срезает
/// частоты выше нового Найквиста и не вносит алиасинга — для речи это
/// заметно улучшает качество распознавания Whisper.
pub fn resample_to_whisper(samples: &[i16], from_rate: u32) -> Vec<f32> {
    // Если частота уже совпадает — просто конвертируем без сэмплинга.
    if from_rate as f64 == TARGET_RATE {
        return samples.iter().map(|&s| i16_to_f32(s)).collect();
    }

    let src: Vec<f32> = samples.iter().map(|&s| i16_to_f32(s)).collect();
    resample_sinc(&src, from_rate as f64, TARGET_RATE)
}

/// Оконно-sinc пересэмплер: для каждого выходного сэмпла суммирует вклад
/// ближайших входных, взвешенный низкочастотным ядром.
fn resample_sinc(src: &[f32], from: f64, to: f64) -> Vec<f32> {
    let ratio = to / from;
    // При понижении частоты нужно срезать всё выше нового Найквиста (to/2):
    // масштабируем ядро, чтобы нули sinc совпадали с сэмплами нового ряда.
    let cutoff_scale = (1.0_f64 / ratio).max(1.0);
    let gain = 1.0 / cutoff_scale;
    let n_out = (src.len() as f64 * ratio).floor() as usize;

    let mut out = Vec::with_capacity(n_out);
    let src_len = src.len() as isize;

    for i in 0..n_out {
        let center = i as f64 / ratio;
        let first = ((center - SINC_TAPS).ceil() as isize).max(0);
        let last = ((center + SINC_TAPS).floor() as isize + 1).min(src_len);

        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        for j in first..last {
            let dist = j as f64 - center;
            let w = sinc(dist / cutoff_scale) * kaiser_window(dist) * gain;
            acc += src[j as usize] as f64 * w;
            wsum += w;
        }

        out.push(if wsum.abs() > 1e-9 {
            (acc / wsum) as f32
        } else {
            0.0
        });
    }

    out
}

/// Обрезает молчание в начале и в конце записи.
///
/// Порог — относительный (5% от самого громкого окна), поэтому не зависит
/// от уровня микрофона и не «вырезает» тихую речь. Сверху порог ограничен
/// абсолютной величиной (-48 дБ), чтобы на громких записях не срезать тихие
/// окончания слов (последний слог «дела» тише гласных середины, но это ещё
/// речь); по бокам оставляется запас в одно окно. Если запись почти
/// полностью состоит из тишины или речь занимает меньше пятой части —
/// ничего не обрезаем, чтобы не испортить короткий клип.
pub fn trim_silence(audio: &[f32]) -> &[f32] {
    const MIN_SPEECH_FRACTION: usize = 5; // речь должна занимать >= 1/5 клипа
    // Запас по одному окну (20 мс) с каждой стороны границы речи: эти окна
    // почти наверняка содержат хвост последней гласной или призвук начала.
    const EDGE_PAD_WINDOWS: usize = 1;

    let rms = window_rms(audio);
    let windows = rms.len();
    if windows < 4 {
        return audio;
    }

    let max_rms = rms.iter().fold(0.0f32, |acc, &x| acc.max(x));
    // Почти всё тишина (микрофон не пишет) — не трогаем запись.
    if max_rms < SILENCE_RMS {
        return audio;
    }

    let threshold = (max_rms * 0.05).min(0.004);
    let mut start = 0;
    while start < windows && rms[start] < threshold {
        start += 1;
    }
    let mut end = windows;
    while end > start && rms[end - 1] < threshold {
        end -= 1;
    }

    // Речь должна занимать хотя бы пятую часть клипа, иначе это короткий
    // отрывок — лучше отдать весь кусок целиком.
    if end - start >= windows / MIN_SPEECH_FRACTION {
        let start = start.saturating_sub(EDGE_PAD_WINDOWS) * WINDOW;
        let end = (end + EDGE_PAD_WINDOWS).min(windows) * WINDOW;
        &audio[start..end]
    } else {
        audio
    }
}

/// Подавляет фоновый шум в записи (мягкое шумоподавление).
///
/// 1. Фильтр высоких частот первого порядка (срез ~100 Гц) убирает
///    постоянную составляющую и низкочастотный гул, не задевая голос.
/// 2. Шумовой гейт по окнам 20 мс: уровень шума оценивается по самым тихим
///    окнам, а окна тише порога приглушаются. Усиление между окнами
///    плавно интерполируется, поэтому на границах не появляется щелчков,
///    а сама речь (громче порога) не искажается.
pub fn apply_noise_reduction(audio: &mut [f32]) {
    if audio.is_empty() {
        return;
    }

    // alpha = 1 - exp(-2π·100/16000): частота среза ~100 Гц при 16 кГц.
    const HIGH_PASS_ALPHA: f32 = 0.0385;
    let mut low = 0.0f32;
    for sample in audio.iter_mut() {
        low += HIGH_PASS_ALPHA * (*sample - low);
        *sample -= low;
    }

    // Уровень по окнам (последнее неполное окно тоже учитываем).
    let rms: Vec<f32> = audio
        .chunks(WINDOW)
        .map(|seg| (seg.iter().map(|x| x * x).sum::<f32>() / seg.len() as f32).sqrt())
        .collect();

    // Уровень шума — 25-й перцентиль RMS окон; порог в 3 раза выше.
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let noise_floor = sorted[sorted.len() / 4];
    let gate = (noise_floor * 3.0).max(0.001);

    // Окна тише порога приглушаются тем сильнее, чем они тише (до MIN_GAIN).
    const MIN_GAIN: f32 = 0.1;
    let gains: Vec<f32> = rms
        .iter()
        .map(|&level| {
            if level >= gate {
                1.0
            } else {
                let ratio = level / gate;
                MIN_GAIN + (1.0 - MIN_GAIN) * ratio * ratio
            }
        })
        .collect();

    // Плавная интерполяция усиления между центрами соседних окон.
    let last = gains.len() - 1;
    for (i, sample) in audio.iter_mut().enumerate() {
        let pos = i as f32 / WINDOW as f32 - 0.5;
        let k0 = if pos <= 0.0 {
            0
        } else {
            (pos.floor() as usize).min(last)
        };
        let k1 = (k0 + 1).min(last);
        let frac = (pos - k0 as f32).clamp(0.0, 1.0);
        *sample *= gains[k0] * (1.0 - frac) + gains[k1] * frac;
    }
}

/// Нормализует громкость: если запись тихая, усиливает её.
///
/// Whisper заметно лучше распознаёт речь нормального уровня. Усиление
/// ограничено, а почти пустые записи не усиливаются, чтобы не поднимать шум.
pub fn normalize_audio(audio: &mut [f32]) {
    if audio.is_empty() {
        return;
    }

    // Средний RMS — если запись почти целиком тишина, не трогаем её.
    let sum: f32 = audio.iter().map(|x| x * x).sum();
    let avg_rms = (sum / audio.len() as f32).sqrt();
    if avg_rms < 0.003 {
        return;
    }

    let peak = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if peak <= 0.0 {
        return;
    }

    // Целевой пик ~ -6 дБ; усиление до 8x (~ +18 дБ). Тихая, но реальная речь
    // (например, при «Усилении входа» ниже 1.0 или слабом микрофоне) должна
    // дойти до Whisper нормальным уровнем, а не «шёпотом», на котором модель
    // выдумывает текст вместо распознавания.
    let target_peak: f32 = 0.5;
    let gain = (target_peak / peak).clamp(1.0, 8.0);

    if (gain - 1.0).abs() < 0.05 {
        return;
    }

    for sample in audio.iter_mut() {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
}

/// Есть ли в записи заметная энергия/речь?
///
/// Тихие и «пустые» записи (быстрое нажатие клавиши, фоновый шум, глубокий
/// тишина) не содержат речи, но Whisper на них умеет «галлюцинировать»
/// (например, «Продолжение следует»). Эта функция помогает отличить
/// реальную речь от тишины: если максимум по окнам RMS ниже порога —
/// речь не считаем распознанной.
pub fn has_speech_energy(audio: &[f32]) -> bool {
    if audio.is_empty() {
        return false;
    }

    if audio.len() < WINDOW {
        // Слишком короткий клип — оцениваем по всему буферу целиком.
        let sum: f32 = audio.iter().map(|x| x * x).sum();
        return (sum / audio.len() as f32).sqrt() >= SILENCE_RMS;
    }

    // Как и в trim_silence: ниже этого RMS считаем запись «тишиной».
    window_rms(audio).into_iter().any(|rms| rms >= SILENCE_RMS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_rate_when_same() {
        let input = vec![0i16, 1000, -2000, 3000];
        let out = resample_to_whisper(&input, 16000);
        assert_eq!(out.len(), input.len());
        assert_eq!(out[0], 0.0);
        assert!((out[1] - 1000.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn resample_down_by_half() {
        // 32 кГц -> 16 кГц: выход вдвое короче входа.
        let input: Vec<i16> = (0..1000i16).collect();
        let out = resample_to_whisper(&input, 32000);
        assert_eq!(out.len(), 500);
    }

    #[test]
    fn trim_only_silent_edges() {
        let mut audio = vec![0.0f32; 3200]; // 10 окон по 20 мс
        for s in &mut audio[800..2400] {
            *s = 0.5; // «речь» в середине
        }
        let trimmed = trim_silence(&audio);
        let non_silent = trimmed.iter().filter(|&&x| x > 0.001).count();
        // Вся «речь» сохранена, с обеих сторон оставлен запас по одному окну
        // (20 мс), чтобы не срезать мягкие окончания слов.
        assert_eq!(non_silent, 1600);
        assert_eq!(trimmed.len(), 2560);
        // Запас захватывает тишину с каждой стороны ровно на одно окно.
        assert_eq!(&trimmed[..320], &audio[320..640]);
    }

    #[test]
    fn normalize_amplifies_quiet() {
        let mut audio = vec![0.1f32, -0.1, 0.2, -0.2];
        normalize_audio(&mut audio);
        let peak = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        assert!(
            (peak - 0.5).abs() < 1e-2,
            "пик должен приблизиться к 0.5, реальный {peak}"
        );
    }

    #[test]
    fn normalize_skips_silence() {
        let mut audio = vec![0.0f32; 100];
        normalize_audio(&mut audio);
        assert!(audio.iter().all(|&x| x == 0.0));
    }

    /// Синусоида заданной частоты и амплитуды (16 кГц).
    fn sine(freq: f32, amplitude: f32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / 16_000.0).sin())
            .collect()
    }

    fn energy(audio: &[f32]) -> f32 {
        audio.iter().map(|x| x * x).sum()
    }

    #[test]
    fn noise_reduction_lowers_quiet_section() {
        // Тихий фон — громкая «речь» — снова тихий фон, по 0.2 с.
        let noise = sine(1000.0, 0.01, 3200);
        let speech = sine(300.0, 0.5, 3200);
        let mut audio = [noise.clone(), speech.clone(), noise.clone()].concat();
        apply_noise_reduction(&mut audio);

        // Окно сразу за речью не учитываем: там затухает «хвост» фильтра.
        let noise_after = energy(&audio[..2880]) + energy(&audio[6720..]);
        let noise_before = energy(&noise[..2880]) + energy(&noise[320..]);
        let speech_after = energy(&audio[3200..6400]);
        assert!(
            noise_after < noise_before * 0.5,
            "шум должен заметно снизиться"
        );
        assert!(
            speech_after > energy(&speech) * 0.8,
            "речь должна сохраниться"
        );
    }

    #[test]
    fn noise_reduction_removes_dc_offset() {
        // Постоянное смещение (DC) — не звук, фильтр должен его убрать.
        let mut audio: Vec<f32> = sine(300.0, 0.3, 6400).iter().map(|x| x + 0.2).collect();
        apply_noise_reduction(&mut audio);
        let tail = &audio[3200..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(mean.abs() < 0.01, "среднее после фильтра: {mean}");
    }

    #[test]
    fn silence_has_no_speech_energy() {
        // Пустая и тихая запись — речи нет (например, быстрое нажатие клавиши).
        assert!(!has_speech_energy(&[]));
        let silent = vec![0.0f32; 6400]; // 400 мс тишины
        assert!(!has_speech_energy(&silent));
    }

    #[test]
    fn loud_audio_has_speech_energy() {
        // Громкая «речь» — энергия есть.
        let mut audio = vec![0.0f32; 6400];
        for s in &mut audio[1600..3200] {
            *s = 0.4;
        }
        assert!(has_speech_energy(&audio));
    }
}
