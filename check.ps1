<#
    Lokalnyi check devaika (dvizhok VoiceAI): fmt + clippy + testy odnoi komandoi.

    Note: file is ASCII-only on purpose. PowerShell 5.1 reads UTF-8 files
    without BOM using the ANSI codepage, which corrupts Cyrillic literals
    and can break parsing. Messages stay English to keep the script portable.

    Usage (Windows PowerShell):
        .\check.ps1

    Runs, in order:
      1. cargo fmt --all -- --check
      2. cargo clippy --all-targets -- -D warnings
      3. cargo test --all-targets

    Exits 0 only when all three steps pass.
#>

$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargo) {
    $fallback = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
    if (Test-Path -LiteralPath $fallback) {
        $cargo = $fallback
    } else {
        Write-Host "cargo not found. Install Rust: https://rustup.rs" -ForegroundColor Red
        exit 1
    }
}

$script:failed = $false

function Run-Step([string]$name, [scriptblock]$body) {
    Write-Host ""
    Write-Host "== $name ==" -ForegroundColor Cyan
    & $body
    if ($LASTEXITCODE -ne 0) {
        $script:failed = $true
    }
}

Run-Step "1/3. Formatting (cargo fmt --check)" {
    & $cargo fmt --all -- --check
}

Run-Step "2/3. Static analysis (cargo clippy -D warnings)" {
    & $cargo clippy --all-targets -- -D warnings
}

Run-Step "3/3. Tests (cargo test --all-targets)" {
    & $cargo test --all-targets
}

Write-Host ""
if ($script:failed) {
    Write-Host "Check finished with errors." -ForegroundColor Red
    exit 1
}
Write-Host "Done: formatting, clippy and tests passed." -ForegroundColor Green
exit 0