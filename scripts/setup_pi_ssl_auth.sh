#!/usr/bin/env bash
set -euo pipefail

# === LawnBerry Pi SSL + Auth Setup ===
# Target: kp@192.168.1.95

echo "=== Setting up SSL certificate ==="
sudo mkdir -p /etc/lawnberry/certs/selfsigned

# Check if cert already exists
if [ ! -f /etc/lawnberry/certs/selfsigned/fullchain.pem ]; then
    echo "Generating self-signed certificate for 192.168.1.95..."
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048         -keyout /etc/lawnberry/certs/selfsigned/privkey.pem \
        -out /etc/lawnberry/certs/selfsigned/fullchain.pem \
        -subj "/CN=192.168.1.95/O=LawnBerry Pi/C=US" \
        -addext "subjectAltName=IP:192.168.1.95,DNS:lawnberry.local"
    sudo chmod 600 /etc/lawnberry/certs/selfsigned/privkey.pem
    sudo chmod 644 /etc/lawnberry/certs/selfsigned/fullchain.pem
    echo "SSL certificate created."
else
    echo "SSL certificate already exists at /etc/lawnberry/certs/selfsigned/"
fi

echo ""
echo "=== Setting up authorized SSH keys ==="
# Ensure .ssh directory exists
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add current user's key if not already present
if [ -f ~/.ssh/authorized_keys ]; then
    echo "authorized_keys already exists."
else
    echo "Creating authorized_keys..."
    touch ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi

# Generate SSH key if none exists
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "Generating SSH key..."
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "kp@192.168.1.95"
    cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
    echo "SSH key generated and added to authorized_keys."
fi

echo ""
echo "=== Verifying nginx configuration ==="
if command -v nginx >/dev/null 2>&1; then
    echo "nginx is installed."
    # Check if lawnberry config exists
    if [ ! -f /etc/nginx/sites-enabled/lawnberry-https ]; then
        echo "Running HTTPS setup script..."
        cd /home/kp/lawnberry-pi 2>/dev/null || cd /home/kp/thordrive/lawnberry-pi 2>/dev/null || echo "WARNING: Could not find lawnberry-pi directory"
        if [ -f scripts/setup_https.sh ]; then
            sudo bash scripts/setup_https.sh
        fi
    fi
else
    echo "nginx not installed. Install with: sudo apt-get install -y nginx"
fi

echo ""
echo "=== Setup complete ==="
echo "SSL cert: /etc/lawnberry/certs/selfsigned/"
echo "Access at: https://192.168.1.95/"
echo "SSH key: ~/.ssh/id_ed25519"
