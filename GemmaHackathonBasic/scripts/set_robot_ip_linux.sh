#!/usr/bin/env bash
# Usage: sudo bash scripts/set_robot_ip_linux.sh eth0     (find the name with: ip link)
IF=${1:-eth0}
nmcli con add type ethernet ifname "$IF" con-name kinova ipv4.method manual ipv4.addresses 192.168.1.11/24 || true
nmcli con up kinova
ping -c 3 192.168.1.10 && echo "OK - open http://192.168.1.10 (admin/admin)"
