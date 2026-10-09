param([string]$Path, [int]$Start=1, [int]$End=99999)
$root='D:\viettel\microservice'
$i=0
Get-Content -LiteralPath (Join-Path $root $Path) -Encoding UTF8 | ForEach-Object { $i++; if($i -ge $Start -and $i -le $End) { '{0}: {1}' -f $i,$_ } }
