# Push EmotionDetection-BE and EmotionDetection-FE to syedahinamukhtar-dev on GitHub.
# All commits are already rewritten to: Syeda Hina Mukhtar <288886451+syedahinamukhtar-dev@users.noreply.github.com>
#
# One-time setup (as syedahinamukhtar-dev):
#   gh auth login
#
# Then run:
#   .\scripts\push-to-syeda-github.ps1

$ErrorActionPreference = "Stop"
$BeRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$FeRoot = (Resolve-Path (Join-Path $BeRoot "..\EmotionDetection-FE")).Path
$Account = "syedahinamukhtar-dev"

gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Run 'gh auth login' while signed in as $Account first."
}

function Ensure-Remote($RepoPath, $RepoName) {
    $url = "https://github.com/$Account/$RepoName.git"
    Push-Location $RepoPath
    git remote remove syeda 2>$null
    git remote add syeda $url
    $exists = gh repo view "$Account/$RepoName" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating $Account/$RepoName ..."
        gh repo create "$Account/$RepoName" --public --description "FER emotion detection - $RepoName"
    }
    Pop-Location
}

Ensure-Remote $BeRoot "EmotionDetection-BE"
Ensure-Remote $FeRoot "EmotionDetection-FE"

Write-Host "Pushing BE (main)..."
Push-Location $BeRoot
git push -u syeda main --force
Pop-Location

Write-Host "Pushing FE (main + hina-dev)..."
Push-Location $FeRoot
git push -u syeda main --force
git push -u syeda hina-dev --force
Pop-Location

Write-Host ""
Write-Host "Done."
Write-Host "  https://github.com/$Account/EmotionDetection-BE"
Write-Host "  https://github.com/$Account/EmotionDetection-FE"
