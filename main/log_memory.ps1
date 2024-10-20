param (
    [int]$ProcessID,  # Cambiato da PID a ProcessID
    [int]$Interval
)

# Controlla se il processo con il ProcessID specificato esiste
try {
    $process = Get-Process -Id $ProcessID -ErrorAction Stop
} catch {
    Write-Host "Process with ID $ProcessID not found."
    exit 1
}

# Ciclo infinito per stampare l'utilizzo della memoria ogni intervallo specificato
while ($true) {
    try {
        # Ottiene l'uso della memoria fisica (Working Set) del processo in kilobyte
        $memUsage = $process.WorkingSet64 / 1KB
        $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Write-Host "${time}: Memory usage: $([math]::Round($memUsage, 2)) KB"

        # Attendi l'intervallo specificato prima di controllare di nuovo
        Start-Sleep -Seconds $Interval

        # Aggiorna le informazioni sul processo
        $process = Get-Process -Id $ProcessID -ErrorAction Stop
    } catch {
        Write-Host "Process with ID $ProcessID has terminated."
        exit 1
    }
}
