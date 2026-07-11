param(
    [string]$EnvFile = (Join-Path $PSScriptRoot "..\.env")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Environment file does not exist: $EnvFile"
}

$values = @{}
foreach ($line in Get-Content -LiteralPath $EnvFile -Encoding UTF8) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
        continue
    }

    $separator = $trimmed.IndexOf("=")
    if ($separator -lt 1) {
        continue
    }

    $name = $trimmed.Substring(0, $separator).Trim()
    $value = $trimmed.Substring($separator + 1).Trim()
    if (
        ($value.StartsWith('"') -and $value.EndsWith('"')) -or
        ($value.StartsWith("'") -and $value.EndsWith("'"))
    ) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    $values[$name] = $value
}

function Get-FirstConfiguredValue {
    param([string[]]$Names)

    foreach ($name in $Names) {
        if ($values.ContainsKey($name) -and $values[$name]) {
            return $values[$name]
        }
    }
    return $null
}

$apiKey = Get-FirstConfiguredValue @("LLM_API_KEY", "DEEPSEEK_API_KEY")
$baseUrl = Get-FirstConfiguredValue @(
    "ANTHROPIC_BASE_URL",
    "DEEPSEEK_ANTHROPIC_BASE_URL"
)
$model = Get-FirstConfiguredValue @("LLM_MODEL_NAME", "DEEPSEEK_MODEL")
$cheapModel = Get-FirstConfiguredValue @("CHEAP_LLM_MODEL_NAME")

if (-not $apiKey) {
    throw "Missing LLM_API_KEY or DEEPSEEK_API_KEY in $EnvFile"
}
if (-not $baseUrl) {
    throw "Missing ANTHROPIC_BASE_URL or DEEPSEEK_ANTHROPIC_BASE_URL in $EnvFile"
}
if (-not $model) {
    throw "Missing LLM_MODEL_NAME or DEEPSEEK_MODEL in $EnvFile"
}
if (-not $cheapModel) {
    $cheapModel = $model
}

$env:PYTHONUTF8 = "1"
$env:LLM_PROVIDER = "anthropic"
$env:LLM_API_KEY = $apiKey
$env:LLM_MODEL_NAME = $model
$env:CHEAP_LLM_MODEL_NAME = $cheapModel
$env:ANTHROPIC_BASE_URL = $baseUrl

Write-Host "Official evaluation environment activated in this PowerShell session."
Write-Host "Provider: $env:LLM_PROVIDER"
Write-Host "Model: $env:LLM_MODEL_NAME"
Write-Host "Cheap model: $env:CHEAP_LLM_MODEL_NAME"
Write-Host "Anthropic base URL: $env:ANTHROPIC_BASE_URL"
Write-Host "API key: <set>"
