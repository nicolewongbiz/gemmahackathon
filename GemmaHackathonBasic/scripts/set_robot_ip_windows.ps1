# Give the laptop's Ethernet adapter a static IP on the Kinova's network.
# Run PowerShell AS ADMINISTRATOR:   .\scripts\set_robot_ip_windows.ps1 -Adapter "Ethernet"
# List adapter names with:           Get-NetAdapter
param([string]$Adapter = "Ethernet", [string]$Ip = "192.168.1.11")
Write-Host "Adapters:"; Get-NetAdapter | Format-Table Name, InterfaceDescription, Status
netsh interface ip set address name="$Adapter" static $Ip 255.255.255.0
Start-Sleep 3
Write-Host "`nPinging the arm at 192.168.1.10 ..."
ping 192.168.1.10 -n 3
Write-Host "`nIf ping works, open http://192.168.1.10 (Kinova Web App, admin/admin)."
Write-Host "To undo later: netsh interface ip set address name=`"$Adapter`" dhcp"
