#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Coffee Grind Analyser Bot — Proxmox LXC Deployment Script
#
# Usage:  ./deploy-proxmox.sh <TELEGRAM_BOT_TOKEN> [BRANCH]
#
# Examples:
#   ./deploy-proxmox.sh "123456:ABC-DEF..."          # deploys from main
#   ./deploy-proxmox.sh "123456:ABC-DEF..." bot       # deploys from 'bot' branch
#
# Run this ON your Proxmox host (as root).  It will:
#   1. Download the Ubuntu 24.04 LXC template (if not cached)
#   2. Create a lightweight LXC container (512 MB RAM, 4 GB disk)
#   3. Install Python, clone the repo, install dependencies
#   4. Create a systemd service that auto-starts the bot
#
# To remove:  pct stop <CTID> && pct destroy <CTID>
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — adjust these if needed
# ---------------------------------------------------------------------------
BRIDGE="vmbr0"                          # Proxmox network bridge
STORAGE="local-lvm"                     # Storage for container rootfs
TEMPLATE_STORAGE="local"                # Storage for ISO/template cache
TEMPLATE="ubuntu-24.04-standard_24.04-2_amd64.tar.zst"
RAM_MB=512
SWAP_MB=256
DISK_GB=4
CORES=2
HOSTNAME="coffee-bot"
REPO_URL="https://github.com/jantielens/coffee-grind-size-analyzer.git"
APP_DIR="/opt/coffee-bot"

# ---------------------------------------------------------------------------
# Validate input
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <TELEGRAM_BOT_TOKEN> [BRANCH]"
    echo ""
    echo "  BRANCH defaults to 'main' if not specified."
    echo "  Get a token from @BotFather on Telegram."
    exit 1
fi

BOT_TOKEN="$1"
BRANCH="${2:-main}"

if [[ ! "$BOT_TOKEN" =~ ^[0-9]+:.+$ ]]; then
    echo "ERROR: Token doesn't look valid (expected format: 123456:ABC-DEF...)"
    exit 1
fi

# Check we're on a Proxmox host
if ! command -v pct &>/dev/null; then
    echo "ERROR: 'pct' not found. Run this script on your Proxmox host."
    exit 1
fi

# ---------------------------------------------------------------------------
# Find next free CTID (100+)
# ---------------------------------------------------------------------------
find_free_ctid() {
    local ctid=100
    local used
    used=$(pct list 2>/dev/null | awk 'NR>1 {print $1}')
    while echo "$used" | grep -qw "$ctid"; do
        ctid=$((ctid + 1))
    done
    echo "$ctid"
}

CTID=$(find_free_ctid)
echo "============================================================"
echo "  Coffee Grind Analyser Bot — Proxmox Deployment"
echo "============================================================"
echo "  CTID:      $CTID"
echo "  Hostname:  $HOSTNAME"
echo "  RAM:       ${RAM_MB} MB"
echo "  Disk:      ${DISK_GB} GB"
echo "  Cores:     $CORES"
echo "  Bridge:    $BRIDGE"
echo "  Storage:   $STORAGE"
echo "  Branch:    $BRANCH"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Download template if not already cached
# ---------------------------------------------------------------------------
echo ">>> Step 1/5: Checking LXC template..."
if pveam list "$TEMPLATE_STORAGE" 2>/dev/null | grep -q "$TEMPLATE"; then
    echo "    Template already cached."
else
    echo "    Downloading $TEMPLATE..."
    pveam download "$TEMPLATE_STORAGE" "$TEMPLATE"
fi

TEMPLATE_PATH="${TEMPLATE_STORAGE}:vztmpl/${TEMPLATE}"

# ---------------------------------------------------------------------------
# Step 2: Create the LXC container
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 2/5: Creating LXC container ($CTID)..."
pct create "$CTID" "$TEMPLATE_PATH" \
    --hostname "$HOSTNAME" \
    --memory "$RAM_MB" \
    --swap "$SWAP_MB" \
    --cores "$CORES" \
    --rootfs "${STORAGE}:${DISK_GB}" \
    --net0 "name=eth0,bridge=${BRIDGE},ip=dhcp" \
    --unprivileged 1 \
    --features nesting=1 \
    --start 0 \
    --onboot 1

echo "    Container $CTID created."

# ---------------------------------------------------------------------------
# Step 3: Start container and wait for network
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 3/5: Starting container..."
pct start "$CTID"

echo "    Waiting for network..."
for i in $(seq 1 30); do
    if pct exec "$CTID" -- ping -c1 -W1 8.8.8.8 &>/dev/null; then
        echo "    Network is up."
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "ERROR: Container has no network after 30s."
        echo "Check your bridge ($BRIDGE) and DHCP setup."
        echo "To clean up:  pct stop $CTID && pct destroy $CTID"
        exit 1
    fi
    sleep 1
done

# ---------------------------------------------------------------------------
# Step 4: Install dependencies and clone repo
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 4/5: Installing Python and cloning repo..."

pct exec "$CTID" -- bash -c "
    set -euo pipefail

    # Install system packages
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv git libgl1 libglib2.0-0 > /dev/null

    # Clone repo
    git clone --depth 1 --branch $BRANCH $REPO_URL $APP_DIR

    # Create venv and install deps
    python3 -m venv ${APP_DIR}/bot/.venv
    ${APP_DIR}/bot/.venv/bin/pip install --no-cache-dir -q -r ${APP_DIR}/bot/requirements.txt

    echo 'Dependencies installed.'
"

# ---------------------------------------------------------------------------
# Step 5: Create systemd service
# ---------------------------------------------------------------------------
echo ""
echo ">>> Step 5/5: Creating systemd service..."

pct exec "$CTID" -- bash -c "
    cat > /etc/systemd/system/coffee-bot.service << 'UNIT'
[Unit]
Description=Coffee Grind Analyser Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}/bot
Environment=COFFEE_BOT_TOKEN=${BOT_TOKEN}
ExecStart=${APP_DIR}/bot/.venv/bin/python ${APP_DIR}/bot/bot.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

    systemctl daemon-reload
    systemctl enable coffee-bot
    systemctl start coffee-bot
"

# Give it a moment to start
sleep 3

# Check status
if pct exec "$CTID" -- systemctl is-active coffee-bot &>/dev/null; then
    STATUS="✓ RUNNING"
else
    STATUS="✗ NOT RUNNING (check logs below)"
fi

echo ""
echo "============================================================"
echo "  Deployment complete!"
echo "============================================================"
echo "  Container:  $CTID ($HOSTNAME)"
echo "  Status:     $STATUS"
echo ""
echo "  Useful commands (run on Proxmox host):"
echo "    pct exec $CTID -- journalctl -u coffee-bot -f    # tail logs"
echo "    pct exec $CTID -- systemctl restart coffee-bot    # restart"
echo "    pct exec $CTID -- systemctl stop coffee-bot       # stop"
echo "    pct stop $CTID && pct destroy $CTID               # remove entirely"
echo ""
echo "  To update the bot:"
echo "    pct exec $CTID -- bash -c 'cd $APP_DIR && git pull'"
echo "    pct exec $CTID -- systemctl restart coffee-bot"
echo "============================================================"

# Show recent logs if not running
if [[ "$STATUS" == *"NOT RUNNING"* ]]; then
    echo ""
    echo "Recent logs:"
    pct exec "$CTID" -- journalctl -u coffee-bot --no-pager -n 20
fi
