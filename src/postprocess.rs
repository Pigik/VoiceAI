//! Пост-обработка распознанного текста.
//!
//! Модель отдаёт «сырой» текст: со словами-паразитами, повторяющимися
//! словами, без единообразной пунктуации и заглавных букв. Этот модуль
//! приводит его к аккуратному виду целиком локально (без интернета и LLM):
//!
//! 1. Заменяет слово «тире» на знак «—», речевые дроби («три пятых» → «3/5»)
//!    и время («пятнадцать тридцать четыре» → «15:34»).
//! 2. Убирает слова-паразиты и звуки-мычания («ну», «вот», «э-э»).
//! 3. Убирает случайные повторы слов («очень очень» → «очень»).
//! 4. Нормализует пунктуацию: пробелы, повторяющиеся знаки, заглавные буквы
//!    в начале предложений, точка в конце.
//! 5. Делит длинный текст на абзацы.

/// Одиночные слова-паразиты. Сюда попадают только слова, удаление которых
/// не искажает смысл: многозначные слова («знаешь», «слушай», «короче»)
/// сюда не входят — «ты знаешь ответ» не должно превращаться в «ты ответ».
const FILLER_WORDS: &[&str] = &["ну", "вот", "типа", "какбы", "блин"];

/// Многословные паразитические обороты, которые удаляются всегда.
const FILLER_PHRASES: &[&str] = &[
    "ну вот",
    "ну и вот",
    "как бы это",
    "ну типа",
    "так сказать",
    "в общем то",
    "в общем-то",
    "короче говоря",
    "скажем так",
];

/// Обороты, которые бывают и паразитами, и полноценной частью фразы
/// («в общем случае», «как бы ты поступил», «это самое важное»). Их удаляем,
/// только когда они выделены запятой как вводные: «В общем, всё готово».
const COMMA_FILLER_PHRASES: &[&str] = &["в общем", "как бы", "это самое", "так вот"];

/// Отдельные звуки-мычания и междометия, не несущие смысла.
///
/// Однобуквенные «а», «о», «у» сюда не входят: это союз и предлоги
/// («а он остался», «думаю о тебе», «у меня»).
const SOUND_WORDS: &[&str] = &[
    "э", "ээ", "э-э", "аа", "а-а", "уу", "у-у", "м", "мм", "м-м", "хм", "угу", "ага", "оо",
];

/// Примерная длина абзаца в символах, после которой начинается новый.
const PARAGRAPH_CHARS: usize = 360;

/// Короткий последний абзац (меньше стольких символов) приклеивается к предыдущему.
const MIN_LAST_PARAGRAPH_CHARS: usize = 40;

/// Числительные от 0 до 19: слово -> число.
const ONES_AND_TEENS: &[(&str, u32)] = &[
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

/// Знаменатели дробей в той форме, в которой их произносят после
/// числителя «одна» (женский род, ед. ч.: «одна пятая»).
const DENOMINATORS_SINGULAR: &[(&str, u32)] = &[
    ("вторая", 2),
    ("третья", 3),
    ("четвёртая", 4),
    ("четвертая", 4),
    ("пятая", 5),
    ("шестая", 6),
    ("седьмая", 7),
    ("восьмая", 8),
    ("девятая", 9),
    ("десятая", 10),
    ("двенадцатая", 12),
    ("двадцатая", 20),
];

/// Знаменатели дробей после остальных числителей (родительный падеж мн. ч.:
/// «три пятых»).
///
/// Порядковые числительные в других формах («двадцать пятое мая»,
/// «двадцать вторая неделя») дробями не считаются.
const DENOMINATORS_PLURAL: &[(&str, u32)] = &[
    ("вторых", 2),
    ("третьих", 3),
    ("четвёртых", 4),
    ("четвертых", 4),
    ("пятых", 5),
    ("шестых", 6),
    ("седьмых", 7),
    ("восьмых", 8),
    ("девятых", 9),
    ("десятых", 10),
    ("двенадцатых", 12),
    ("двадцатых", 20),
];

/// Сигналы, что перед нами время, а не число: слова «час», «часа», «часов».
fn is_hour_word(word: &str) -> bool {
    matches!(word, "час" | "часа" | "часов")
}

/// Слова, обозначающие минуты в речевом времени.
fn is_minute_word(word: &str) -> bool {
    matches!(word, "минута" | "минуты" | "минут")
}

/// Парсит русское числительное (одно слово) в число.
fn parse_number_word(word: &str) -> Option<u32> {
    let low = lowercase_word(word);
    ONES_AND_TEENS
        .iter()
        .chain(TENS)
        .find(|(name, _)| *name == low)
        .map(|&(_, value)| value)
}

/// Парсит числитель дроби из одного-двух слов: «три» -> 3,
/// «двадцать одна» -> 21. Другие сочетания («три два») — не число.
fn parse_compound_number(words: &[&str]) -> Option<u32> {
    match words {
        [single] => parse_number_word(single),
        [tens, ones] => {
            let tens = parse_number_word(tens)?;
            let ones = parse_number_word(ones)?;
            (tens >= 20 && tens % 10 == 0 && (1..=9).contains(&ones)).then_some(tens + ones)
        }
        _ => None,
    }
}

/// Нужен ли после числителя знаменатель в единственном числе
/// («одна пятая», «двадцать одна пятая»), а не во множественном («две пятых»).
fn takes_singular_denominator(numerator: u32) -> bool {
    numerator % 10 == 1 && numerator % 100 != 11
}

/// Заменяет слово «тире» на знак тире.
fn replace_dash_word(tokens: &[String]) -> Vec<String> {
    tokens
        .iter()
        .map(|token| {
            let stripped = strip_trailing_punct(token);
            if lowercase_word(stripped) == "тире" {
                format!("\u{2014}{}", trailing_punct(token))
            } else {
                token.clone()
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
        // Числитель — одно-два слова-числа подряд без знаков препинания
        // между ними (иначе это уже разные части фразы).
        let mut num_words: Vec<&str> = Vec::new();
        let mut j = i;
        while j < tokens.len() && num_words.len() < 2 {
            let token = &tokens[j];
            if !trailing_punct(token).is_empty() || parse_number_word(token).is_none() {
                break;
            }
            num_words.push(token.as_str());
            j += 1;
        }

        if j < tokens.len()
            && let Some(numerator) = parse_compound_number(&num_words)
        {
            let denom_low = lowercase_word(strip_trailing_punct(&tokens[j]));
            let table = if takes_singular_denominator(numerator) {
                DENOMINATORS_SINGULAR
            } else {
                DENOMINATORS_PLURAL
            };
            if let Some(&(_, denominator)) = table.iter().find(|(name, _)| *name == denom_low) {
                result.push(format!(
                    "{numerator}/{denominator}{}",
                    trailing_punct(&tokens[j])
                ));
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

/// Заменяет речевое время на формат ЧЧ:ММ.
///
/// Конвертация происходит только при явном признаке времени, чтобы не
/// искажать диктуемые числа и номера:
///   «пятнадцать тридцать четыре» -> «15:34»
///   «15 34» -> «15:34»
///   «час тридцать» -> «01:30»
///   «три часа сорок пять» -> «03:45»
///   «двенадцать ноль ноль» -> «12:00»
///
/// Число «двадцать три» при этом не превращается в «20:03», а слова,
/// разделённые знаком препинания, не склеиваются в одно время.
fn replace_time_expressions(tokens: &[String]) -> Vec<String> {
    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let token = &tokens[i];
        let low = lowercase_word(token);

        // Кандидат в часы — слово «час» (= 1), число цифрами или словом.
        // Токен со знаком препинания («один.») — конец фразы, а не время.
        let mut hour_val: Option<u32> = None;
        let mut strong = false;
        if trailing_punct(token).is_empty() {
            if is_hour_word(&low) {
                hour_val = Some(1);
                strong = true;
            } else if !low.is_empty()
                && low.chars().all(|c| c.is_ascii_digit())
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
        }

        if let Some(h) = hour_val {
            let mut j = i + 1;
            let mut stopped_by_punct = false;
            // Пропускаем слова «час»/«часа»/«часов» — верный признак времени.
            while j < tokens.len()
                && !stopped_by_punct
                && is_hour_word(&lowercase_word(strip_trailing_punct(&tokens[j])))
            {
                strong = true;
                stopped_by_punct = !trailing_punct(&tokens[j]).is_empty();
                j += 1;
            }

            // Собираем число минут (до 3 токенов), пропуская слова «минут...».
            // Токен со знаком препинания завершает время.
            let mut parts: Vec<u32> = Vec::new();
            while j < tokens.len() && parts.len() < 3 && !stopped_by_punct {
                let s = strip_trailing_punct(&tokens[j]);
                let l = lowercase_word(s);
                if is_minute_word(&l) {
                    strong = true;
                } else if let Some(n) = parse_number_word(&l).or_else(|| s.parse::<u32>().ok()) {
                    parts.push(n);
                } else {
                    break;
                }
                stopped_by_punct = !trailing_punct(&tokens[j]).is_empty();
                j += 1;
            }

            let compound_minutes = parts.len() >= 2;
            if let Some(m) = compound_from_parts(&parts)
                && m <= 59
                && (strong || compound_minutes && h >= 10)
            {
                result.push(format!("{h:02}:{m:02}{}", trailing_punct(&tokens[j - 1])));
                i = j;
                continue;
            }
        }

        result.push(token.clone());
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

/// Открывающие скобки и кавычки: после них не ставится пробел, а первая
/// буква предложения за ними всё равно становится заглавной.
fn is_opening(c: char) -> bool {
    matches!(c, '(' | '[' | '«' | '„' | '"' | '\'')
}

fn lowercase_word(word: &str) -> String {
    word.chars()
        .map(|c| c.to_lowercase().next().unwrap_or(c))
        .collect()
}

/// Отделяет от слова завершающую пунктуацию («привет,» -> «привет»).
/// Если слово целиком состоит из знаков, возвращается как есть.
fn strip_trailing_punct(word: &str) -> &str {
    let stripped = word.trim_end_matches(|c: char| is_punct(c) || matches!(c, '"' | '\'' | ')'));
    if stripped.is_empty() { word } else { stripped }
}

/// Завершающая пунктуация слова («привет,» -> «,»).
fn trailing_punct(word: &str) -> &str {
    &word[strip_trailing_punct(word).len()..]
}

/// Приводит слово к «словарному» виду для сравнения: только буквы, цифры
/// и дефисы, в нижнем регистре.
fn keep_word_chars(word: &str) -> String {
    word.chars()
        .filter(|&c| c.is_alphanumeric() || c == '-' || c == '\u{2014}')
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

/// Не теряет конец предложения при удалении паразита: «Привет, ну.» ->
/// «Привет.», а не «Привет,».
fn keep_sentence_end(result: &mut [String], removed_last: &str) {
    let ends: String = trailing_punct(removed_last)
        .chars()
        .filter(|&c| is_sentence_end(c))
        .collect();
    if ends.is_empty() {
        return;
    }
    if let Some(prev) = result.last_mut() {
        let trimmed_len = prev.trim_end_matches([',', ';', ':']).len();
        prev.truncate(trimmed_len);
        if !prev.ends_with(is_sentence_end) {
            prev.push_str(&ends);
        }
    }
}

/// Удаляет слова-паразиты и повторы слов.
///
/// Идём по словам слева направо: если цепочка слов совпадает с одним из
/// паразитических оборотов (или одно слово — известный паразит/звук), она
/// пропускается. Также вырезаются случайные повторы одного и того же слова,
/// характерные для устной речи («я я», «очень очень»).
fn remove_fillers(tokens: &[String]) -> Vec<String> {
    // (слова оборота, удалять только если оборот выделен запятой)
    let mut phrases: Vec<(Vec<String>, bool)> = FILLER_PHRASES
        .iter()
        .map(|p| (split_words(p), false))
        .chain(COMMA_FILLER_PHRASES.iter().map(|p| (split_words(p), true)))
        .collect();
    // Длинные обороты проверяем раньше, чтобы «ну и вот» распознавался как
    // один оборот, а не как «ну» + «вот».
    phrases.sort_by_key(|(words, _)| std::cmp::Reverse(words.len()));

    let mut result: Vec<String> = Vec::with_capacity(tokens.len());
    let n = tokens.len();
    let mut i = 0;

    while i < n {
        let stripped = strip_trailing_punct(&tokens[i]);
        let low = keep_word_chars(stripped);

        // Многословный оборот-паразит.
        let matched = phrases.iter().find_map(|(words, needs_comma)| {
            let len = words.len();
            if i + len > n {
                return None;
            }
            let same = words.iter().enumerate().all(|(k, needle)| {
                keep_word_chars(strip_trailing_punct(&tokens[i + k])) == *needle
            });
            let comma_ok = !needs_comma || trailing_punct(&tokens[i + len - 1]).contains(',');
            (same && comma_ok).then_some((len, *needs_comma))
        });
        if let Some((len, parenthetical)) = matched {
            // Вводный оборот «, как бы,» убираем вместе с обрамляющими запятыми.
            if parenthetical && let Some(prev) = result.last_mut() {
                let trimmed_len = prev.trim_end_matches(',').len();
                prev.truncate(trimmed_len);
            }
            keep_sentence_end(&mut result, &tokens[i + len - 1]);
            i += len;
            continue;
        }

        // Одиночный паразит или звук-междометие.
        if FILLER_WORDS.contains(&low.as_str()) || is_sound_word(stripped) {
            keep_sentence_end(&mut result, &tokens[i]);
            i += 1;
            continue;
        }

        // Случайный повтор слова подряд (включая «я я», «он он»).
        let prev = result
            .last()
            .map(|w| keep_word_chars(strip_trailing_punct(w)));
        if !low.is_empty() && prev.as_deref() == Some(low.as_str()) {
            keep_sentence_end(&mut result, &tokens[i]);
            i += 1;
            continue;
        }

        result.push(tokens[i].clone());
        i += 1;
    }

    result
}

/// Является ли «-» в позиции `idx` тире (отбитым пробелом справа), а не
/// дефисом внутри слова («кто-то», «из-за») или минусом («-5»).
fn is_standalone_hyphen(chars: &[char], idx: usize) -> bool {
    chars.get(idx + 1).is_none_or(|c| c.is_whitespace())
}

/// Нормализует пунктуацию и регистр во всей строке.
///
/// - схлопывает лишние пробелы и убирает пробелы перед знаками пунктуации;
/// - схлопывает повторяющиеся знаки («,,», «!!»);
/// - отбивает тире пробелами, не трогая дефисы внутри слов;
/// - ставит заглавную букву в начале каждого предложения;
/// - добавляет точку в конец, если её нет.
///
/// Прочие символы (скобки, кавычки, «%», «+», «=» и т.п.) сохраняются
/// вместе с исходными пробелами вокруг них.
fn normalize_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len() + 4);
    let mut prev: char = '\0';
    let mut gap = false;
    let mut sentence_start = true;

    for (idx, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            gap = true;
            continue;
        }

        if is_punct(c) {
            // Повторяющиеся одинаковые знаки пунктуации схлопываем.
            if c != prev {
                remove_trailing_space(&mut out);
                out.push(c);
                if is_sentence_end(c) {
                    sentence_start = true;
                }
            }
        } else if c == '\u{2014}' || (c == '-' && is_standalone_hyphen(&chars, idx)) {
            if !out.is_empty() && out.last() != Some(&' ') {
                out.push(' ');
            }
            out.push(c);
            // После тире всегда пробел (модель пишет «— текст»).
            out.push(' ');
        } else {
            let after_space_or_open = out.is_empty()
                || gap
                || out.last() == Some(&' ')
                || out.last().is_some_and(|&last| is_opening(last));
            if gap && out.last().is_some_and(|&last| last != ' ') {
                out.push(' ');
            }
            // Заглавная — только в начале слова: «т.д.» и «3.14» не трогаем.
            if sentence_start && c.is_alphabetic() && after_space_or_open {
                out.extend(c.to_uppercase());
            } else {
                out.push(c);
            }
            if c.is_alphanumeric() {
                sentence_start = false;
            }
        }
        prev = c;
        gap = false;
    }

    remove_trailing_space(&mut out);

    // Текст не закончился знаком конца предложения — ставим точку
    // (висящие «,», «;», «:» в конце заменяем на неё).
    if out.iter().any(|c| c.is_alphanumeric()) {
        while out.last().is_some_and(|&c| matches!(c, ',' | ';' | ':')) {
            out.pop();
        }
        if !out.last().is_some_and(|&c| is_sentence_end(c)) {
            out.push('.');
        }
    }

    out.into_iter().collect()
}

fn remove_trailing_space(out: &mut Vec<char>) {
    while out.last() == Some(&' ') {
        out.pop();
    }
}

/// Разбивает текст на предложения по знакам конца предложения, за которыми
/// идёт пробел (точка внутри «3.14» или «т.д.» предложение не делит).
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        current.push(c);
        let at_boundary = chars.peek().is_none_or(|next| next.is_whitespace());
        if is_sentence_end(c) && at_boundary {
            let s = current.trim();
            if !s.is_empty() {
                sentences.push(s.to_string());
            }
            current.clear();
        }
    }

    let tail = current.trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }

    sentences
}

/// Группирует предложения в абзацы.
///
/// Абзац заканчивается, когда накоплено примерно `PARAGRAPH_CHARS` символов.
/// Слишком короткий «хвостовой» абзац присоединяется к предыдущему, чтобы
/// текст не выглядел рваным.
fn group_paragraphs(sentences: &[String]) -> Vec<String> {
    let mut paragraphs: Vec<String> = Vec::new();
    let mut current = String::new();

    for sentence in sentences {
        let candidate_len = current.chars().count() + sentence.chars().count();
        if !current.is_empty() && candidate_len > PARAGRAPH_CHARS {
            paragraphs.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(sentence);
    }
    if !current.is_empty() {
        paragraphs.push(current);
    }

    // Короткий последний абзац приклеиваем к предыдущему.
    if paragraphs.len() >= 2
        && paragraphs
            .last()
            .is_some_and(|last| last.chars().count() < MIN_LAST_PARAGRAPH_CHARS)
        && let Some(last) = paragraphs.pop()
        && let Some(prev) = paragraphs.last_mut()
    {
        prev.push(' ');
        prev.push_str(&last);
    }

    paragraphs
}

/// Главная функция пост-обработки: сырой текст модели -> аккуратный текст.
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

    // В тексте не осталось ни букв, ни цифр (например, были только звуки
    // и знаки пунктуации) — такой текст не несёт смысла.
    if !joined.chars().any(|c| c.is_alphanumeric()) {
        return String::new();
    }

    if !auto_punctuation {
        return joined;
    }

    let normalized = normalize_punctuation(&joined);
    let sentences = split_sentences(&normalized);
    group_paragraphs(&sentences).join("\n\n")
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
        assert_eq!(pp("ну я блин готов"), "Я готов.");
    }

    #[test]
    fn keeps_meaningful_verbs_and_comparatives() {
        assert_eq!(pp("ты знаешь ответ"), "Ты знаешь ответ.");
        assert_eq!(pp("слушай меня внимательно"), "Слушай меня внимательно.");
        assert_eq!(pp("этот путь короче"), "Этот путь короче.");
    }

    #[test]
    fn removes_sound_words() {
        assert_eq!(pp("э-э я мм думаю"), "Я думаю.");
        assert_eq!(pp("угу ага конечно"), "Конечно.");
    }

    #[test]
    fn keeps_conjunction_and_prepositions() {
        // «а», «о», «у» — союз и предлоги, а не мычание.
        assert_eq!(pp("я пошёл а он остался"), "Я пошёл а он остался.");
        assert_eq!(pp("думаю о тебе"), "Думаю о тебе.");
        assert_eq!(pp("у меня дома"), "У меня дома.");
        assert_eq!(pp("у тебя всё хорошо"), "У тебя всё хорошо.");
        assert_eq!(pp("книга у меня"), "Книга у меня.");
    }

    #[test]
    fn removes_repeated_words() {
        assert_eq!(pp("очень очень интересно"), "Очень интересно.");
        assert_eq!(pp("я я согласен"), "Я согласен.");
        assert_eq!(pp("он он придёт"), "Он придёт.");
    }

    #[test]
    fn removes_phrases() {
        assert_eq!(pp("так сказать это, как бы, секрет"), "Это секрет.");
        assert_eq!(pp("ну и вот начинаем"), "Начинаем.");
        assert_eq!(pp("ну вот начинаем"), "Начинаем.");
        assert_eq!(pp("В общем, всё готово"), "Всё готово.");
    }

    #[test]
    fn keeps_ambiguous_phrases_without_comma() {
        assert_eq!(pp("в общем случае решение"), "В общем случае решение.");
        assert_eq!(pp("как бы ты поступил?"), "Как бы ты поступил?");
        assert_eq!(pp("это самое важное"), "Это самое важное.");
    }

    #[test]
    fn filler_at_sentence_end_keeps_the_period() {
        assert_eq!(pp("привет, ну. как дела"), "Привет. Как дела.");
        assert_eq!(pp("готово, короче говоря!"), "Готово!");
    }

    #[test]
    fn adds_capital_and_period() {
        assert_eq!(pp("привет мир"), "Привет мир.");
        assert_eq!(pp("один. два три"), "Один. Два три.");
        assert_eq!(pp("Привет МИР"), "Привет МИР.");
        assert_eq!(pp("привет мир,"), "Привет мир.");
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
        assert_eq!(pp("я тире студент"), "Я — студент.");
    }

    #[test]
    fn keeps_hyphenated_words() {
        assert_eq!(
            pp("кто-то пришёл из-за дождя"),
            "Кто-то пришёл из-за дождя."
        );
        assert_eq!(pp("на улице -5 градусов"), "На улице -5 градусов.");
    }

    #[test]
    fn keeps_numbers_abbreviations_and_symbols() {
        assert_eq!(pp("число 3.14 важно"), "Число 3.14 важно.");
        assert_eq!(pp("яблоки, груши и т.д."), "Яблоки, груши и т.д.");
        assert_eq!(pp("скидка 50% (примерно)"), "Скидка 50% (примерно).");
        assert_eq!(pp("2+2=4"), "2+2=4.");
        assert_eq!(pp("«война и мир» роман"), "«Война и мир» роман.");
    }

    #[test]
    fn converts_fractions() {
        assert_eq!(pp("одна вторая"), "1/2.");
        assert_eq!(pp("две третьих и три пятых"), "2/3 и 3/5.");
        assert_eq!(pp("двадцать одна пятая"), "21/5.");
    }

    #[test]
    fn ordinals_are_not_fractions() {
        assert_eq!(pp("двадцать пятое мая"), "Двадцать пятое мая.");
        assert_eq!(pp("двадцать вторая неделя"), "Двадцать вторая неделя.");
    }

    #[test]
    fn converts_time() {
        assert_eq!(
            pp("встреча в пятнадцать тридцать четыре"),
            "Встреча в 15:34."
        );
        assert_eq!(pp("в три часа сорок пять"), "В 03:45.");
        assert_eq!(pp("в пятнадцать тридцать четыре?"), "В 15:34?");
        assert_eq!(pp("их было двадцать три"), "Их было двадцать три.");
    }

    #[test]
    fn time_does_not_cross_punctuation() {
        assert_eq!(
            pp("было двенадцать. тридцать пять человек пришли"),
            "Было двенадцать. Тридцать пять человек пришли."
        );
    }

    #[test]
    fn splits_paragraphs() {
        let long = (1..=6u32)
            .map(|i| {
                format!(
                    "Предложение номер {i} двухтысячного года является очень длинным и \
                     очень содержательным."
                )
            })
            .collect::<Vec<_>>()
            .join(" ");
        let result = pp(&long);
        assert!(
            result.contains("\n\n"),
            "длинный текст должен делиться на абзацы: {result}"
        );
        for paragraph in result.split("\n\n") {
            assert!(paragraph.chars().count() <= PARAGRAPH_CHARS + 100);
        }
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
