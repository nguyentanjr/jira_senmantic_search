$script:Root = 'D:\viettel\microservice'
function Show-Code([string]$Path, [int]$Start=1, [int]$End=10000) {
 $full=Join-Path $script:Root $Path
 $lines=[System.IO.File]::ReadAllLines($full)
 $Path
 for($i=$Start-1;$i -lt [Math]::Min($End,$lines.Length);$i++){ '{0}: {1}' -f ($i+1),$lines[$i] }
}
function Find-Code([string]$Pattern,[string]$Repo='',[string]$Filter='*.java') {
 $dir=Join-Path $script:Root $Repo
 Get-ChildItem -LiteralPath $dir -Recurse -File -Filter $Filter | Select-String -Pattern $Pattern | ForEach-Object { '{0}:{1}: {2}' -f $_.Path.Substring($script:Root.Length+1),$_.LineNumber,$_.Line.Trim() }
}

