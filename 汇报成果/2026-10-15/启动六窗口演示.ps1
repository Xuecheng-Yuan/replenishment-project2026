$ErrorActionPreference='Stop'
$taskRuntime=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$taskHelper=Join-Path $PSScriptRoot '启动六窗口.mjs'
if(Test-Path -LiteralPath $taskRuntime){
    Start-Process -FilePath $taskRuntime -ArgumentList ('"'+$taskHelper+'"') -WindowStyle Hidden
}else{
    Start-Process -FilePath (Join-Path $PSScriptRoot '动画\index.html')
    Write-Host '在页面点击“打开六个同步窗口”。若浏览器拦截弹出窗口，请允许本演示打开窗口。'
}
