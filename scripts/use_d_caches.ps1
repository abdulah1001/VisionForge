$ErrorActionPreference = "Continue"

# VisionForge: pin all download/cache paths to D: (C: has no free space).
$CacheRoot = "D:\caches"
@(
    "pip",
    "huggingface",
    "huggingface\hub",
    "huggingface\transformers",
    "torch",
    "tmp",
    "xdg",
    "sam3_src"
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot $_) | Out-Null
}

$env:PIP_CACHE_DIR = "$CacheRoot\pip"
$env:HF_HOME = "$CacheRoot\huggingface"
$env:HUGGINGFACE_HUB_CACHE = "$CacheRoot\huggingface\hub"
$env:TRANSFORMERS_CACHE = "$CacheRoot\huggingface\transformers"
$env:TORCH_HOME = "$CacheRoot\torch"
$env:XDG_CACHE_HOME = "$CacheRoot\xdg"
$env:TEMP = "$CacheRoot\tmp"
$env:TMP = "$CacheRoot\tmp"
$env:TMPDIR = "$CacheRoot\tmp"
$env:PIP_TMPDIR = "$CacheRoot\tmp"

# Junction common C: cache paths onto D: so tools that ignore env vars still stay off C:.
$links = @{
    "$env:LOCALAPPDATA\pip\cache" = "$CacheRoot\pip"
    "$env:USERPROFILE\.cache\huggingface" = "$CacheRoot\huggingface"
    "$env:USERPROFILE\.cache\torch" = "$CacheRoot\torch"
}
foreach ($src in $links.Keys) {
    $dst = $links[$src]
    $parent = Split-Path $src -Parent
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if (Test-Path $src) {
        $item = Get-Item $src -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            continue
        }
        Write-Host "Moving existing cache $src -> $dst"
        robocopy $src $dst /E /MOVE /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        if (Test-Path $src) { Remove-Item -Recurse -Force $src -ErrorAction SilentlyContinue }
    }
    if (-not (Test-Path $src)) {
        cmd /c "mklink /J `"$src`" `"$dst`"" | Out-Null
    }
}

Write-Host "VisionForge caches redirected to $CacheRoot"
Get-PSDrive C, D | Format-Table Name, @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}
