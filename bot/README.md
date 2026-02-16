# Coffee Grind Size Analyser — Telegram Bot

A Telegram bot that analyses coffee grind photos on the reference sheet and returns a full pipeline summary.

## Deployment

### Option A — Proxmox LXC (recommended for always-on)

One command on your Proxmox host creates an LXC container, installs everything, and starts the bot as a systemd service:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/jantielens/coffee-grind-size-analyzer/main/bot/deploy-proxmox.sh)" -- "YOUR_BOT_TOKEN"
```

To deploy from a specific branch:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/jantielens/coffee-grind-size-analyzer/main/bot/deploy-proxmox.sh)" -- "YOUR_BOT_TOKEN" bot
```

This creates a lightweight Ubuntu 24.04 LXC (512 MB RAM, 4 GB disk) with auto-start on boot and automatic restart on crash.

**Managing the container** (run on Proxmox host):

| Task | Command |
|------|---------|
| Tail logs | `pct exec <CTID> -- journalctl -u coffee-bot -f` |
| Restart bot | `pct exec <CTID> -- systemctl restart coffee-bot` |
| Update code | `pct exec <CTID> -- bash -c 'cd /opt/coffee-bot && git pull'` then restart |
| Remove entirely | `pct stop <CTID> && pct destroy <CTID>` |

> **Note:** The default config uses `vmbr0` as the network bridge and `local-lvm` for storage. If your Proxmox setup differs, download the script and edit the variables at the top before running.

---

### Option B — Manual setup (local / any Linux)

#### 1. Create a Telegram Bot

1. Open Telegram and chat with [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the bot token (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

#### 2. Install Dependencies

```bash
cd bot
pip install -r requirements.txt
```

#### 3. Set the Token

```bash
# Linux / macOS
export COFFEE_BOT_TOKEN="your-token-here"

# Windows (PowerShell)
$env:COFFEE_BOT_TOKEN = "your-token-here"

# Windows (cmd)
set COFFEE_BOT_TOKEN=your-token-here
```

#### 4. Run the Bot

```bash
python bot.py
```

## Usage

1. Open Telegram and start a chat with your bot
2. Send `/start` to see the welcome message
3. Send a photo of coffee grounds on the printed reference sheet
4. Wait a few seconds — the bot will reply with:
   - `summary.png` — full pipeline visualisation (histogram, gauges, stats, processing steps, overlay)
   - A text caption with key metrics (estimated grind setting, particle count, median diameter, D10/D50/D90)

## Architecture

The bot imports the analysis pipeline directly from `src/`:

```
bot/bot.py  ──imports──►  src/analyze.py
                          src/constants.py
                          src/detection.py
                          src/preprocessing.py
                          src/segmentation.py
                          src/measurement.py
                          src/visualization.py
```

No changes are made to the existing `src/` code. The bot calls the same functions as `python analyze.py --report`.

## Tips for Best Results

- Use your phone's **flash** for consistent lighting
- Keep all **4 ArUco markers** fully visible
- Spread grounds in a **thin, even layer** on the white area
- Hold the camera **20–30 cm** above the sheet, roughly perpendicular
