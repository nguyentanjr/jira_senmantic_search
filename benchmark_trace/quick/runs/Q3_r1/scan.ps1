param([string]$Pattern, [string]$Repo='', [string]$Glob='*.java', [int]$Context=0)
$root='D:\viettel\microservice'
$base=Join-Path $root $Repo
Get-ChildItem -LiteralPath $base -Recurse -File -Filter $Glob | Where-Object { $_.FullName -notmatch '\\(target|\.git)\\' } | Select-String -Pattern $Pattern -Context $Context,$Context | ForEach-Object {
  $relative=$_.Path.Substring($root.Length+1)
  if($Context -gt 0) { $_.Context.PreContext }
  '{0}:{1}: {2}' -f $relative,$_.LineNumber,$_.Line.TrimEnd()
  if($Context -gt 0) { $_.Context.PostContext }
}
