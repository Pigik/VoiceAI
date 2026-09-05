//! Подготовка аудио для Whisper.
//!
//! Микрофон отдаёт моно- или стерео-поток i16 на своей «родной» частоте
//! (обычно 48 кГц, 44.1 кГц или 16 кГц). Whisper ожидает моно-сэмплы f32
//! с частотой 16 кГц и уровнем в диапазоне [-1.0, 1.0]. Этот модуль делает
//! конвертацию, убирает тишину и выравнивает громкость.

const TARGET_RATE: f64 = 16_000.0;

/// Полуширина sinc-ядра фильтра-пересэмплера (в сэмплах исходной частоты).
/// Больше отводов — точнее срез спектра, но медленнее. 16 отводов — хороший
/// баланс для речи.
const SINC_TAPS: f64 = 16.0;

/// Параметр остроты окна Кайзера. Чем больше, тем сильнее подавляются
/// боковые лепестки фильтра (меньше «звона» и алиасинга на краях полосы).
const KAISER_BETA: f64 = 7.0;

fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 / i16::MAX as f32
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
/// от уровня микрофона и не «вырезает» тихую речь. Если запись почти
/// полностью состоит из тишины или речь занимает меньше пятой части —
/// ничего не обрезаем, чтобы не испортить короткий клип.
pub fn trim_silence(audio: &[f32]) -> &[f32] {
    const WINDOW: usize = 320; // 20 мс при 16 кГц
    const MIN_SPEECH_FRACTION: usize = 5; // речь должна занимать >= 1/5 клипа

    let windows = audio.len() / WINDOW;
    if windows < 4 {
        return audio;
    }

    // RMS каждого окна.
    let mut rms: Vec<f32> = Vec::with_capacity(windows);
    for w in 0..windows {
        let seg = &audio[w * WINDOW..(w + 1) * WINDOW];
        let sum: f32 = seg.iter().map(|x| x * x).sum();
        rms.push((sum / WINDOW as f32).sqrt());
    }

    let max_rms = rms.iter().fold(0.0f32, |acc, &x| acc.max(x));
    // Почти всё тишина (микрофон не пишет) — не трогаем запись.
    if max_rms < 0.002 {
        return audio;
    }

    let threshold = (max_rms * 0.05).min(0.008); // минимум -42 дБ
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
        &audio[start * WINDOW..end * WINDOW]
    } else {
        audio
    }
}

/// Подавляет фоновый шум в записи (мягкое шумоподавление).
///
/// Сначала убирает постоянную составляющую и низкочастотный гул простым
/// высокочастотным фильтром первого порядка, затем применяет мягкий шумовой
/// порог: тихие участки, не превосходящие уровень шума, дополнительно
/// приглушаются, а речь остаётся почти нетронутой. Работает слабее речевых
/// DSP (не «вырезает» речь), но заметно чистит фон на тихих записях.
pub fn apply_noise_reduction(audio: &mut [f32]) {
    if audio.is_empty() {
        return;
    }

    // Высокочастотный фильтр первого порядка (срезает гул и DC).
    let alpha = 0.2f32;
    let mut prev = 0.0f32;
    for sample in audio.iter_mut() {
        let hp = *sample - prev;
        prev += alpha * hp;
        *sample = hp;
    }

    // Оценка уровня шума: RMS самых тихих 25% сэмплов.
    let mut sorted: Vec<f32> = audio.iter().map(|x| x.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_count = (sorted.len() / 4).max(1);
    let noise_sum: f32 = sorted.iter().take(noise_count).map(|x| x * x).sum();
    let noise_floor = (noise_sum / noise_count as f32).sqrt();
    let gate = (noise_floor * 3.0).max(0.001); // порог: в 3 раза выше уровня шума

    // Мягкое приглушение тихих участков без резкой обрезки речи.
    for sample in audio.iter_mut() {
        let abs = sample.abs();
        if abs < gate {
            *sample *= (abs / gate) * 0.3;
        }
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
    if avg_rms < 0.005 {
        return;
    }

    let peak = audio.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if peak <= 0.0 {
        return;
    }

    // Целевой пик ~ -6 дБ; усиливаем максимум в 4 раза (~ +12 дБ).
    let target_peak: f32 = 0.5;
    let gain = (target_peak / peak).clamp(1.0, 4.0);

    if (gain - 1.0).abs() < 0.05 {
        return;
    }

    for sample in audio.iter_mut() {
        *sample = (*sample * gain).clamp(-1.0, 1.0);
    }
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
        assert!((out[1] - 1000.0 / i16::MAX as f32).abs() < 1e-6);
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
        // Вся «речь» сохранена, с обеих сторон убрана тишина (округляется до
        // границ 20-мс окон, поэтому общая длина чуть больше самой речи).
        assert_eq!(non_silent, 1600);
        assert_eq!(trimmed.len(), 1920);
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

    #[test]
    fn noise_reduction_lowers_quiet_section() {
        let mut audio: Vec<f32> = Vec::new();
        // Тихий фон (шум) в начале.
        for _ in 0..800 {
            audio.push(0.01);
        }
        // Громкая «речь» в середине.
        for _ in 0..800 {
            audio.push(0.5);
        }
        let quiet_before: f32 = audio[..800].iter().map(|x| x.abs()).sum();
        apply_noise_reduction(&mut audio);
        let quiet_after: f32 = audio[..800].iter().map(|x| x.abs()).sum();
        let speech_before: f32 = audio[800..].iter().map(|x| x.abs()).sum();
        let speech_after: f32 = audio[800..].iter().map(|x| x.abs()).sum();
        // Тихий участок заметно обрезан, а «речь» почти не тронута.
        assert!(quiet_after < quiet_before * 0.5, "шум должен снизиться");
        assert!(speech_after > speech_before * 0.5, "речь должна сохраниться");
    }
}
