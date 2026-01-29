# test_api.ps1 내용 (간소화)
$url = "http://localhost:5000/nn_predict"
$body = @{
    q = @(0.1, 0.2, 0.3)
    refl = @(0.9, 0.8, 0.7) # 필수 키 포함
} | ConvertTo-Json

try {
    $res = Invoke-RestMethod -Method Post -Uri $url -Body $body -ContentType "application/json"
    Write-Host "✅ API 성공! Loss:" $res.std_loss
} catch {
    Write-Host "❌ API 실패:" $_.Exception.Message
}