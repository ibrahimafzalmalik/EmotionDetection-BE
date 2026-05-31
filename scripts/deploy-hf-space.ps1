# Deploy EmotionDetection-BE to a Hugging Face Docker Space.
# Prerequisites: git, git-lfs, hf CLI (hf auth login).
#
#   .\scripts\deploy-hf-space.ps1 -HfUser YOUR_USERNAME
#   .\scripts\deploy-hf-space.ps1 -HfUser YOUR_USERNAME -SpaceName fer-emotion-api

param(
    [Parameter(Mandatory = $true)]
    [string]$HfUser,
    [string]$SpaceName = "fer-emotion-api",
    [string]$WorkDir = ""
)

$ErrorActionPreference = "Stop"
$BeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $WorkDir) {
    $WorkDir = Join-Path $env:TEMP "hf-space-$SpaceName"
}

$remote = "https://huggingface.co/spaces/$HfUser/$SpaceName"
$cloneUrl = "$remote.git"

Write-Host "BE root:  $BeRoot"
Write-Host "HF Space: $remote"
Write-Host "Work dir: $WorkDir"

hf auth whoami 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Not logged in. Run: hf auth login  (token needs write scope)"
}

if (Test-Path $WorkDir) {
    if (Test-Path (Join-Path $WorkDir ".git")) {
        Push-Location $WorkDir
        git pull --rebase 2>$null
        Pop-Location
    } else {
        Remove-Item -Recurse -Force $WorkDir
    }
}

if (-not (Test-Path $WorkDir)) {
    git clone $cloneUrl $WorkDir 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating Docker Space '$SpaceName'..."
        hf repo create $SpaceName --type space --space_sdk docker
        if ($LASTEXITCODE -ne 0) { throw "hf repo create failed" }
        git clone $cloneUrl $WorkDir
        if ($LASTEXITCODE -ne 0) { throw "git clone failed after create" }
    }
}

Push-Location $WorkDir
git lfs install 2>$null | Out-Null

# Clean old app files (keep .git)
Get-ChildItem -Force | Where-Object { $_.Name -ne ".git" } | Remove-Item -Recurse -Force

Copy-Item -Force (Join-Path $BeRoot "Dockerfile") .
Copy-Item -Force (Join-Path $BeRoot "README.md") .
if (Test-Path (Join-Path $BeRoot ".gitattributes")) {
    Copy-Item -Force (Join-Path $BeRoot ".gitattributes") .
}

# backend/ (no __pycache__)
robocopy (Join-Path $BeRoot "backend") "backend" /E /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null

# fer_project/ inference-only
New-Item -ItemType Directory -Force -Path "fer_project" | Out-Null
foreach ($f in @("config.py", "__init__.py", "README.md")) {
    Copy-Item -Force (Join-Path $BeRoot "fer_project\$f") "fer_project\"
}
robocopy (Join-Path $BeRoot "fer_project\models") "fer_project\models" /E /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
robocopy (Join-Path $BeRoot "fer_project\outputs") "fer_project\outputs" /E /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null

git add -A
if (-not (git status --porcelain)) {
    Write-Host "Already up to date."
    Pop-Location
    Write-Host "App URL: https://${HfUser}-${SpaceName}.hf.space"
    exit 0
}

git commit -m "Deploy FER FastAPI Docker backend"
Write-Host "Pushing (LFS may take several minutes)..."
git push
if ($LASTEXITCODE -ne 0) {
    Write-Host @"

Push failed? Try:
  hf auth login
  git lfs install
  Git Xet on PATH if you see git-xet: command not found
  https://huggingface.co/docs/hub/xet/using-xet-storage#git
"@
    Pop-Location
    exit 1
}

Pop-Location
Write-Host ""
Write-Host "Deployed. Watch build: $remote"
Write-Host "Health: https://${HfUser}-${SpaceName}.hf.space/health"
Write-Host "Vercel: NEXT_PUBLIC_API_URL=https://${HfUser}-${SpaceName}.hf.space"
