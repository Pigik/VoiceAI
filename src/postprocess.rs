//! Пост-обработка распознанного текста.
//!
//! Whisper отдаёт «сырой» текст: со словами-паразитами, повторяющимися
//! словами, без единообразной пунктуации и заглавных букв. Этот модуль
//! приводит его к аккуратному виду целиком локально (без интернета и LLM):
//!
//! 1. Убирает слова-паразиты и звуки-мычания («ну», «вот», «как бы», «э-э»).
//! 2. Убирает случайные повторы слов («очень очень» -> «очень»).
//! 3. Нормализует пунктуацию (пробелы, повторяющиеся знаки, заглавные буквы
//!    в начале предложений, точка в конце).
//! 4. Делит длинный текст на абзацы.

/// Одиночные слова-паразиты. Подобраны так, чтобы удаление не искажало
/// смысл: это классические разговорные «пустышки» без собственного значения.
const FILLER_WORDS: &[&str] = &[
    "ну",
    "вот",
    "типа",
    "какбы",
    "знаешь",
    "короче",
    "прикинь",
    "блин",
    "слушай",
    "слушайте",
];

/// Многословные паразитические обороты (ищутся целиком, длинные — раньше).
const FILLER_PHRASES: &[&str] = &[
    "ну вот",
    "ну и вот",
    "это самое",
    "как бы это",
    "ну типа",
    "так сказать",
    "в общем то",
    "короче говоря",
    "как бы",
    "в общем",
    "так вот",
    "скажем так",
];

/// Отдельные звуки-мычания и междометия, не несущие смысла.
const SOUND_WORDS: &[&str] = &[
    "э", "ээ", "э-э", "а", "аа", "а-а", "уу", "у-у", "м", "мм", "м-м", "хм", "угу", "ага", "о",
    "оо",
];

/// Верхняя граница «длины» абзаца в символах, после которой начинается новый.
const PARAGRAPH_CHARS: usize = 360;

/// Словарь числителительных: слово -> цифра (0..19).
const ONES_TWENTIES: &[(&str, u32)] = &[
    ("ноль", 0),
    ("нуль", 0),
    ("один", 1),
    ("одна", 1),
    ("одно", 1),
    ("два", 2),
    ("две", 2),
    ("три", 3),
    ("четыре", 4),
    ("пять", 5),
    ("шесть", 6),
    ("семь", 7),
    ("восемь", 8),
    ("девять", 9),
    ("десять", 10),
    ("одиннадцать", 11),
    ("двенадцать", 12),
    ("тринадцать", 13),
    ("четырнадцать", 14),
    ("пятнадцать", 15),
    ("шестнадцать", 16),
    ("семнадцать", 17),
    ("восемнадцать", 18),
    ("девятнадцать", 19),
];

/// Десятки: «двадцать» -> 20, «тридцать» -> 30 и т.д.
const TENS: &[(&str, u32)] = &[
    ("двадцать", 20),
    ("тридцать", 30),
    ("сорок", 40),
    ("пятьдесят", 50),
    ("шестьдесят", 60),
    ("семьдесят", 70),
    ("восемьдесят", 80),
    ("девяносто", 90),
];

/// Знаменатели дробей: «пятых» -> 5, «десятых» -> 10 и т.д.
const DENOMINATORS: &[(&str, u32)] = &[
    ("вторая", 2),
    ("вторых", 2),
    ("второе", 2),
    ("третья", 3),
    ("третьих", 3),
    ("третье", 3),
    ("четвёртая", 4),
    ("четвертая", 4),
    ("четвёртых", 4),
    ("четвертых", 4),
    ("четвёртое", 4),
    ("четвертое", 4),
    ("пятая", 5),
    ("пятых", 5),
    ("пятое", 5),
    ("шестая", 6),
    ("шестых", 6),
    ("шестое", 6),
    ("седьмая", 7),
    ("седьмых", 7),
    ("седьмое", 7),
    ("восьмая", 8),
    ("восьмых", 8),
    ("восьмое", 8),
    ("девятая", 9),
    ("девятых", 9),
    ("девятое", 9),
    ("десятая", 10),
    ("десятых", 10),
    ("десятое", 10),
    ("двенадцатая", 12),
    ("двенадцатых", 12),
    ("двенадцатое", 12),
    ("двадцатая", 20),
    ("двадцатых", 20),
    ("двадцатое", 20),
];

/// Сигналы, что перед нами время, а не число: слова «час», «часа», «часов».
fn is_hour_word(word: &str) -> bool {
    matches!(word, "час" | "часа" | "часов")
}

/// Слова, обозначающие минуты в речевом времени.
fn is_minute_word(word: &str) -> bool {
    matches!(word, "минута" | "минуты" | "минут")
}

/// Парсит русское числительное (одно слово) в число. Возвращает None, если слово не является числом.
fn parse_number_word(word: &str) -> Option<u32> {
    let low = lowercase_word(word);
    for &(name, val) in ONES_TWENTIES {
        if low == name {
            return Some(val);
        }
    }
    for &(name, val) in TENS {
        if low == name {
            return Some(val);
        }
    }
    None
}

/// Парсит составное числительное из нескольких слов (до 3): «триста двадцать один» -> 321.
fn parse_compound_number(words: &[&str]) -> Option<u32> {
    let mut total: u32 = 0;
    let mut current: u32 = 0;
    for w in words {
        if let Some(v) = parse_number_word(w) {
            if v >= 100 {
                if current == 0 {
                    current = v;
                } else {
                    current *= v;
                }
            } else if v >= 10 {
                if current == 0 {
                    current = v;
                } else {
                    current += v;
                }
            } else {
                current += v;
            }
        } else {
            return None;
        }
    }
    total = if current > 0 { current } else { total };
    if total > 0 { Some(total) } else { None }
}

/// Заменяет слово «тире» на символ «-» в токенах.
fn replace_dash_word(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .map(|t| {
            let stripped = strip_trailing_punct(t);
            let low = lowercase_word(stripped);
            if low == "тире" {
                "-".to_string()
            } else {
                t.clone()
            }
        })
        .collect()
}

/// Заменяет речевые дроби на запись «числитель/знаменатель».
/// Например: «три пятых» -> «3/5», «одна вторая» -> «1/2».
fn replace_fractions(tokens: &[String]) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        // Ищем конструкцию: <числительное> <знаменатель>
        // Числительное может быть составным: «триста двадцать одна пятая».
        // Пока берём простой случай: 1-2 слова числительного + 1 слово знаменателя.
        let mut num_words: Vec<&str> = Vec::new();
        let mut j = i;
        while j < tokens.len() {
            let s = strip_trailing_punct(&tokens[j]);
            let l = lowercase_word(s);
            if parse_number_word(&l).is_some() {
                num_words.push(s);
                j += 1;
            } else {
                break;
            }
        }

        if j < tokens.len() && !num_words.is_empty() {
            let denom_stripped = strip_trailing_punct(&tokens[j]);
            let denom_low = lowercase_word(denom_stripped);
            if let Some(&(_, denom_val)) = DENOMINATORS.iter().find(|&&(name, _)| denom_low == name)
                && let Some(num_val) = parse_compound_number(&num_words)
            {
                result.push(format!("{num_val}/{denom_val}"));
                i = j + 1;
                continue;
            }
        }

        result.push(tokens[i].clone());
        i += 1;
    }
    result
}

/// Складывает части числа: [30, 4] -> 34; [12] -> 12; [0, 5] -> 5.
fn compound_from_parts(parts: &[u32]) -> Option<u32> {
    if parts.is_empty() {
        return None;
    }
    let mut total: u32 = 0;
    for &p in parts {
        if p == 0 {
            continue;
        }
        if p >= 10 {
            total += p;
        } else if total == 0 {
            total = p;
        } else if total >= 10 {
            total += p;
        } else {
            total = total * 10 + p;
        }
    }
    Some(total)
}

/// Заменяет речевое время на формат HH:MM.
///
/// Конвертация происходит только при явном признаке времени, чтобы не
/// искажать диктуемые числа и номера:
///   «пятнадцать тридцать четыре» -> «15:34»
///   «15 34» -> «15:34»
///   «час тридцать» -> «1:30»
///   «три часа сорок пять» -> «3:45»
///   «двенадцать ноль ноль» -> «12:00»
/// Слово «двадцать три» как число при этом не превращается во «20:03».
fn replace_time_expressions(tokens: &[String]) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let stripped = strip_trailing_punct(&tokens[i]);
        let low = lowercase_word(stripped);

        // Час: слово «час»/«часа»/«часов» (=1) либо число. Токен с пунктуацией
        // час не кандидат («один.» после точки — конец предложения).
        let mut hour_val: Option<u32> = None;
        let mut strong = false;
        if is_hour_word(&low) {
            hour_val = Some(1);
            strong = true;
        } else if tokens[i] == stripped
            && low.chars().all(|c| c.is_ascii_digit() || c == ':')
            && let Ok(v) = low.parse::<u32>()
            && v <= 23
        {
            hour_val = Some(v);
            strong = true;
        } else if let Some(v) = parse_number_word(&low)
            && v <= 23
        {
            hour_val = Some(v);
        }

        if let Some(h) = hour_val {
            let mut j = i + 1;
            // Пропускаем слова «час»/«часа»/«часов» — верный признак времени.
            while j < tokens.len()
                && is_hour_word(&lowercase_word(strip_trailing_punct(&tokens[j])))
            {
                strong = true;
                j += 1;
            }

            // Собираем число минут (до 3 токенов), игнорируя слова «минут...».
            let mut parts: Vec<u32> = Vec::new();
            while j < tokens.len() && parts.len() < 3 {
                let s = strip_trailing_punct(&tokens[j]);
                let l = lowercase_word(s);
                if is_minute_word(&l) {
                    strong = true;
                    j += 1;
                    continue;
                }
                match parse_number_word(&l).or_else(|| s.parse::<u32>().ok()) {
                    Some(n) => {
                        parts.push(n);
                        j += 1;
                    }
                    None => break,
                }
            }

            let compound_minutes = parts.len() >= 2;
            if let Some(m) = compound_from_parts(&parts)
                && m <= 59
                && (strong || compound_minutes && h >= 10)
            {
                result.push(format!("{h:02}:{m:02}"));
                i = j;
                continue;
            }
        }

        result.push(tokens[i].clone());
        i += 1;
    }
    result
}

/// Знаки, которые заканчивают предложение.
fn is_sentence_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

/// Знаки пунктуации, перед которыми нужно убрать пробел.
/// «/» нужен, чтобы сохранялись дроби («3/5») и номера.
fn is_punct(c: char) -> bool {
    matches!(c, '.' | ',' | '!' | '?' | ':' | ';' | '/' | '…')
}

/// Тире в предложении (длинное «—» и символ «-», которым мы заменяем слово «тире»).
fn is_dash(c: char) -> bool {
    c == '\u{2014}' || c == '-'
}

fn lowercase_word(word: &str) -> String {
    word.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect()
}

/// Отделяет от слова завершающую пунктуацию («привет,» -> «привет»).
fn strip_trailing_punct(word: &str) -> &str {
    let mut end = word.len();
    while end > 0 {
        let c = word[..end].chars().next_back().unwrap();
        if is_punct(c) || matches!(c, '"' | '\'' | ')') {
            end -= c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 { word } else { &word[..end] }
}

/// Приводит слово к «словарному» виду для сравнения: только буквы, цифры
/// и дефисы, в нижнем регистре.
fn keep_word_chars(word: &str) -> String {
    word.chars()
        .filter(|&c| c.is_alphanumeric() || c == '-' || is_dash(c))
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect()
}

fn is_sound_word(stripped: &str) -> bool {
    SOUND_WORDS.contains(&lowercase_word(stripped).as_str())
}

/// Разбивает сырой текст на слова по пробелам и переносам строк.
fn split_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_string()).collect()
}

/// Удаляет слова-паразиты и повторы слов.
///
/// Идём по словам слева направо: если цепочка слов совпадает с одним из
/// паразитических оборотов (или одно слово — известный паразит/звук), она
/// пропускается. Также вырезаются случайные повторы одного и того же слова,
/// характерные для устной речи («я я», «очень очень»).
fn remove_fillers(tokens: &[String]) -> Vec<String> {
    let mut phrases: Vec<Vec<String>> = FILLER_PHRASES.iter().map(|p| split_words(p)).collect();
    // Длинные обороты проверяем раньше, чтобы «ну и вот» распознавался как
    // один оборот, а не как «ну» + «вот».
    phrases.sort_by_key(|p| std::cmp::Reverse(p.len()));

    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let n = tokens.len();
    let mut i = 0;

    while i < n {
        let stripped = strip_trailing_punct(&tokens[i]);
        let low = keep_word_chars(stripped);

        // Многословный оборот-паразит.
        let mut matched_phrase = false;
        for phrase in &phrases {
            let len = phrase.len();
            if i + len > n {
                continue;
            }
            let same = phrase.iter().enumerate().all(|(k, needle)| {
                let token_key = keep_word_chars(strip_trailing_punct(&tokens[i + k]));
                token_key.as_str() == needle.as_str()
            });
            if same {
                i += len;
                matched_phrase = true;
                break;
            }
        }
        if matched_phrase {
            continue;
        }

        // Одиночный паразит или звук-междометие.
        if FILLER_WORDS.contains(&low.as_str()) || is_sound_word(stripped) {
            i += 1;
            continue;
        }

        // Случайный повтор слова подряд (включая «я я», «он он»).
        let prev = result
            .last()
            .map(|w| keep_word_chars(strip_trailing_punct(w)));
        if prev.as_deref() == Some(low.as_str()) {
            i += 1;
            continue;
        }

        result.push(tokens[i].clone());
        i += 1;
    }

    result
}

/// Нормализует пунктуацию и регистр во всей строке.
///
/// - убирает лишние пробелы и пробелы перед знаками пунктуации;
/// - схлопывает повторяющиеся знаки («,,», «!!»);
/// - оставляет тире с пробелами вокруг;
/// - ставит заглавную букву в начале каждого предложения;
/// - добавляет точку в конец, если её нет.
fn normalize_punctuation(text: &str) -> String {
    let mut out: Vec<char> = Vec::with_capacity(text.len() + 4);
    let mut prev: char = '\0';
    let mut pregap = false;
    let mut sentence_start = true;
    let mut any_word = false;

    for c in text.chars() {
        if c.is_whitespace() {
            pregap = true;
            continue;
        }

        // Повторяющиеся одинаковые знаки пунктуации схлопываем.
        if is_punct(c) && is_punct(prev) && prev == c {
            continue;
        }

        if is_sentence_end(c) {
            remove_trailing_space(&mut out);
            out.push(c);
            prev = c;
            sentence_start = true;
            pregap = false;
            continue;
        }

        if is_punct(c) {
            remove_trailing_space(&mut out);
            out.push(c);
            prev = c;
            pregap = false;
            continue;
        }

        if is_dash(c) {
            if !out.is_empty() && out.last() != Some(&' ') {
                out.push(' ');
            }
            out.push(c);
            // После тире всегда ставим пробел (Whisper пишет «— текст»).
            out.push(' ');
            prev = c;
            sentence_start = false;
            pregap = false;
            any_word = true;
            continue;
        }

        if c.is_alphanumeric() {
            // Пробел перед словом.
            let starts_with_open = matches!(out.last(), Some('(') | Some('"') | Some('\''));
            if pregap && !starts_with_open && out.last() != Some(&' ') && !out.is_empty() {
                out.push(' ');
            }

            let mut ch = c;
            if sentence_start && c.is_alphabetic() {
                let upper = c.to_uppercase().next().unwrap_or(c);
                ch = upper;
            }

            out.push(ch);
            prev = ch;
            sentence_start = false;
            pregap = false;
            any_word = true;
        }
    }

    trim_end_spaces(&mut out);

    // Если текст не закончился знаком конца предложения — ставим точку.
    if any_word && !out.is_empty() && !is_sentence_end(*out.last().unwrap_or(&'.')) {
        out.push('.');
    }

    out.into_iter().collect()
}

fn remove_trailing_space(out: &mut Vec<char>) {
    while out.last() == Some(&' ') {
        out.pop();
    }
}

fn trim_end_spaces(out: &mut Vec<char>) {
    while out.last() == Some(&' ') {
        out.pop();
    }
}

/// Разбивает текст на предложения (по знакам конца предложения).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if is_sentence_end(c) {
            let s = current.trim().to_string();
            if !s.is_empty() {
                sentences.push(s);
            }
            current.clear();
        }
    }

    let tail = current.trim().to_string();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    sentences
}

/// Группирует предложения в абзацы.
///
/// Абзац заканчивается, когда накоплено примерно `PARAGRAPH_CHARS` символов.
/// Слишком короткий «хвостовой» абзац присоединяется к предыдущему, чтобы
/// текст не выглядел рваным.
fn group_paragraphs(sentences: &[String]) -> Vec<String> {
    if sentences.len() <= 1 {
        return sentences.to_vec();
    }

    let mut paragraphs: Vec<String> = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        let candidate_len = current.len() + sentence.len();
        if !current.is_empty() && candidate_len > PARAGRAPH_CHARS {
            paragraphs.push(current.trim().to_string());
            current = sentence.clone();
        } else if current.is_empty() {
            current = sentence.clone();
        } else {
            current.push(' ');
            current.push_str(sentence);
        }
    }

    if !current.trim().is_empty() {
        paragraphs.push(current.trim().to_string());
    }

    // Короткий последний абзац приклеиваем к предыдущему.
    if paragraphs.len() >= 2 {
        let last = paragraphs.pop().unwrap();
        if last.chars().count() < 40 && paragraphs.last_mut().is_some() {
            if let Some(prev) = paragraphs.last_mut() {
                if !prev.is_empty() {
                    prev.push(' ');
                }
                prev.push_str(&last);
            }
        } else {
            paragraphs.push(last);
        }
    }

    paragraphs
}

/// Главная функция пост-обработки: сырой текст из Whisper -> аккуратный текст.
///
/// Если `auto_punctuation == false`, пунктуация не добавляется и не
/// нормализуется: текст остаётся «как сказан», без точек, заглавных букв
/// и разбиения на абзацы (убираются только слова-паразиты и лишние пробелы).
pub fn postprocess_text(raw: &str, auto_punctuation: bool) -> String {
    let tokens = split_words(raw);
    if tokens.is_empty() {
        return String::new();
    }

    let tokens = replace_dash_word(&tokens);
    let tokens = replace_fractions(&tokens);
    let tokens = replace_time_expressions(&tokens);
    let cleaned = remove_fillers(&tokens);
    let joined = cleaned.join(" ");

    if !auto_punctuation {
        let trimmed = joined.trim().to_string();
        // В тексте не осталось ни букв, ни цифр (например, были только звуки
        // и знаки пунктуации) — такой текст не несёт смысла.
        if trimmed.chars().any(|c| c.is_alphanumeric()) {
            trimmed
        } else {
            String::new()
        }
    } else {
        let normalized = normalize_punctuation(&joined);

        // В тексте не осталось ни букв, ни цифр (например, были только звуки
        // и знаки пунктуации) — такой текст не несёт смысла.
        if !normalized.chars().any(|c| c.is_alphanumeric()) {
            return String::new();
        }

        let sentences = split_sentences(&normalized);
        group_paragraphs(&sentences).join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pp(text: &str) -> String {
        postprocess_text(text, true)
    }

    fn pp_raw(text: &str) -> String {
        postprocess_text(text, false)
    }

    #[test]
    fn removes_filler_words() {
        assert_eq!(pp("ну вот привет"), "Привет.");
        assert_eq!(pp("привет ну вот как дела"), "Привет как дела.");
        assert_eq!(pp("это типа классная идея"), "Это классная идея.");
        assert_eq!(pp("ну я короче готов"), "Я готов.");
    }

    #[test]
    fn removes_sound_words() {
        assert_eq!(pp("э-э я мм думаю"), "Я думаю.");
        assert_eq!(pp("угу ага конечно"), "Конечно.");
    }

    #[test]
    fn removes_repeated_words() {
        assert_eq!(pp("очень очень интересно"), "Очень интересно.");
        assert_eq!(pp("я я согласен"), "Я согласен.");
        assert_eq!(pp("он он придёт"), "Он придёт.");
    }

    #[test]
    fn removes_phrases() {
        assert_eq!(pp("так сказать это как бы секрет"), "Это секрет.");
        assert_eq!(pp("ну и вот начинаем"), "Начинаем.");
        assert_eq!(pp("ну вот начинаем"), "Начинаем.");
    }

    #[test]
    fn adds_capital_and_period() {
        assert_eq!(pp("привет мир"), "Привет мир.");
        assert_eq!(pp("один. два три"), "Один. Два три.");
        assert_eq!(pp("Привет МИР"), "Привет МИР.");
    }

    #[test]
    fn collapses_punctuation() {
        assert_eq!(pp("привет,,  мир!!"), "Привет, мир!");
        assert_eq!(pp("привет . мир"), "Привет. Мир.");
        assert_eq!(pp("это!!! отлично"), "Это! Отлично.");
    }

    #[test]
    fn keeps_dash() {
        assert_eq!(pp("это — важно"), "Это — важно.");
        assert_eq!(pp("я — студент"), "Я — студент.");
    }

    #[test]
    fn splits_paragraphs() {
        let long = (1..=6u32)
            .map(|i| {
                format!(
                    "Предложение номер {i} двухтысячного года является очень длинным и очень содержательным."
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let result = pp(&long);
        assert!(
            result.contains("\n\n"),
            "длинный текст должен делиться на абзацы: {result}"
        );
    }

    #[test]
    fn short_text_single_paragraph() {
        assert_eq!(pp("привет мир"), "Привет мир.");
        assert_eq!(pp(""), "");
    }

    #[test]
    fn keeps_meaningful_words() {
        // «там» — осмысленное указание на место, не паразит.
        assert_eq!(pp("оно там лежит"), "Оно там лежит.");
    }

    #[test]
    fn keeps_question() {
        assert_eq!(pp("привет как тебя зовут?"), "Привет как тебя зовут?");
        assert_eq!(pp("привет как тебя зовут"), "Привет как тебя зовут.");
    }

    #[test]
    fn multiline_segments_are_joined() {
        assert_eq!(
            pp("первая строка\nвторая строка"),
            "Первая строка вторая строка."
        );
    }

    #[test]
    fn empty_and_punctuation_only() {
        assert_eq!(pp("..."), "");
        assert_eq!(pp("   "), "");
    }

    #[test]
    fn raw_mode_keeps_text_as_is() {
        // Без автопунктуации: без точки, без заглавной буквы, без абзацев.
        assert_eq!(pp_raw("привет мир"), "привет мир");
        assert_eq!(pp_raw("привет.. как дела"), "привет.. как дела");
    }

    #[test]
    fn raw_mode_still_removes_fillers() {
        assert_eq!(pp_raw("ну вот привет"), "привет");
        assert_eq!(pp_raw("э-э я мм думаю"), "я думаю");
    }

    #[test]
    fn keeps_preposition_u() {
        // Одиночное «у» — это предлог («у меня», «у тебя»), его нельзя вырезать.
        assert_eq!(pp("у меня дома"), "У меня дома.");
        assert_eq!(pp("у тебя всё хорошо"), "У тебя всё хорошо.");
        assert_eq!(pp("книга у меня"), "Книга у меня.");
    }

    #[test]
    fn raw_mode_empty_when_only_sounds() {
        assert_eq!(pp_raw("э-э мм ..."), "");
        assert_eq!(pp_raw("   "), "");
    }

    #[test]
    fn raw_mode_does_not_split_paragraphs() {
        let long = (1..=6u32)
            .map(|i| format!("предложение номер {i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let result = pp_raw(&long);
        assert!(
            !result.contains("\n\n"),
            "без автопунктуации абзацы не создаются: {result}"
        );
    }
}
