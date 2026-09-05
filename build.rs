// Скрипт сборки: автоматически кладёт модель Whisper и библиотеки CUDA рядом
// с исполняемым файлом при каждом cargo build. Благодаря этому программа
// запускается из любой папки даже после `cargo clean` — вручную ничего
// копировать не нужно.
//
// Модель берётся из корня проекта, библиотеки CUDA — из установленного
// CUDA Toolkit. Скрипт выполняется заново только при изменении build.rs,
// поэтому копирование не замедляет повторные сборки.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // Перезапускаем копирование только когда меняется сам скрипт.
    println!("cargo:rerun-if-changed=build.rs");

    // Всё ниже специфично для Windows (CUDA, имена DLL, пути).
    if env::consts::OS != "windows" {
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // OUT_DIR имеет вид <target>/<profile>/build/<crate>-<hash>/out,
    // значит три уровня вверх — это папка, куда ляжет exe
    // (например, target/release или target/debug).
    let Ok(out_dir) = env::var("OUT_DIR") else {
        return;
    };
    let Some(profile_dir) = Path::new(&out_dir).ancestors().nth(3) else {
        return;
    };

    copy_models(&manifest, profile_dir);
    copy_cuda_dlls(profile_dir);
}

/// Копирует модель Whisper из корня проекта рядом с exe.
fn copy_models(manifest: &Path, dest: &Path) {
    let name = "ggml-large-v3-turbo.bin";
    let src = manifest.join(name);
    if src.is_file() {
        let _ = fs::copy(&src, dest.join(name));
    }
}

/// Копирует библиотеки CUDA (cudart/cublas) рядом с exe — без них whisper-rs
/// не сможет загрузить модель на видеокарте.
fn copy_cuda_dlls(dest: &Path) {
    let Some(bin_dir) = find_cuda_bin() else {
        return;
    };

    let prefixes = ["cudart64_1", "cublas64_1", "cublasLt64_1"];
    let Ok(entries) = fs::read_dir(&bin_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".dll") && prefixes.iter().any(|p| name.starts_with(p)) {
            let _ = fs::copy(entry.path(), dest.join(&name));
        }
    }
}

/// Находит папку с CUDA-библиотеками: сначала по переменной CUDA_PATH,
/// затем сканирует стандартную папку установки CUDA Toolkit.
fn find_cuda_bin() -> Option<PathBuf> {
    if let Ok(path) = env::var("CUDA_PATH") {
        let base = PathBuf::from(path);
        if let Some(dir) = first_existing_bin(&base) {
            return Some(dir);
        }
    }

    let base = Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
    let versions: Vec<String> = fs::read_dir(base)
        .ok()?
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with('v'))
        .collect();

    // Выбираем самую свежую версию: v13.3 > v12.8.
    let newest = versions
        .iter()
        .max_by(|a, b| version_key(a).cmp(&version_key(b)))?;
    first_existing_bin(&base.join(newest))
}

fn first_existing_bin(base: &Path) -> Option<PathBuf> {
    for rel in ["bin\\x64", "bin"] {
        let dir = base.join(Path::new(rel));
        if dir.is_dir() {
            return Some(dir);
        }
    }
    None
}

fn version_key(name: &str) -> Vec<u64> {
    name.trim_start_matches('v')
        .split('.')
        .filter_map(|part| part.parse().ok())
        .collect()
}
