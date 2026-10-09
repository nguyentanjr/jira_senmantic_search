$SourceRoot = 'D:\viettel\microservice'
function Find-Code {
  param([string]$Pattern, [string]$Repo = '', [string]$Filter = '*.java')
  $searchRoot = Join-Path $SourceRoot $Repo
  Get-ChildItem -LiteralPath $searchRoot -Recurse -File -Filter $Filter | Select-String -Pattern $Pattern | ForEach-Object { '{0}:{1}: {2}' -f $_.Path.Substring($SourceRoot.Length + 1), $_.LineNumber, $_.Line.Trim() }
}
function Show-Code {
  param([string]$Path, [int]$Start = 1, [int]$End = 100000)
  if (-not [System.IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $SourceRoot $Path }
  $lines = Get-Content -LiteralPath $Path -Encoding UTF8
  $rel = $Path.Substring($SourceRoot.Length + 1)
  $rel
  for ($i = [Math]::Max(1,$Start); $i -le [Math]::Min($End, $lines.Length); $i++) { '{0}: {1}' -f $i, $lines[$i-1] }
}
function Find-File {
  param([string]$Name, [string]$Repo = '')
  Get-ChildItem -LiteralPath (Join-Path $SourceRoot $Repo) -Recurse -File -Filter $Name | ForEach-Object { $_.FullName.Substring($SourceRoot.Length + 1) }
}
