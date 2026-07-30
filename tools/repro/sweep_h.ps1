# Balayage de BQ_CONTACT_BAND_MULT : etancheite contre epaississement apparent.
$src = "C:\Users\nicol\Code\bourrasque_v2\core\src\mlsmpm.cu"
$verre = "C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\verre.py"
$orig = Get-Content $src -Raw

foreach ($h in @("0.5f", "1.0f", "1.5f", "2.0f")) {
    $new = $orig -replace '#define BQ_CONTACT_BAND_MULT [0-9.]+f', "#define BQ_CONTACT_BAND_MULT $h"
    Set-Content -Path $src -Value $new -Encoding utf8 -NoNewline
    Push-Location "C:\Users\nicol\Code\bourrasque_v2\build"
    $b = cmake --build . --config Release 2>&1
    Pop-Location
    if ($LASTEXITCODE -ne 0) { Write-Output "=== h = $h : BUILD ECHOUE ==="; $b | Select-Object -Last 10; continue }
    Write-Output "==================== h = $h * dx ===================="
    python $verre
}

Set-Content -Path $src -Value $orig -Encoding utf8 -NoNewline
Write-Output "source restaure a l'etat initial (h = 0.5f)"
