# Coffee Grind Analyser — MCP Server

Exposes the coffee grind size analysis pipeline as an [MCP](https://modelcontextprotocol.io/) server so AI assistants (Claude, Copilot, etc.) can call it directly.

## Tools

| Tool | Description |
|---|---|
| `analyze_grind_photo` | Full pipeline — send a base64-encoded photo, get back particle stats, estimated grind setting, and brew recommendation |
| `estimate_grind_setting` | Quick lookup — convert a median particle diameter (mm) to an estimated DF54 grind setting |
| `get_brew_recommendation` | Suggest a brew method for a given median particle diameter |

## Resources

| URI | Description |
|---|---|
| `reference://instructions` | How to take a photo for analysis (printable reference sheet link, tips) |

## Setup

```bash
cd mcp_server
pip install -r requirements.txt
```

## Running

### Stdio transport (local — default)

```bash
python server.py
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "coffee-grind-analyzer": {
      "command": "python",
      "args": ["<full-path-to>/mcp_server/server.py"]
    }
  }
}
```

### VS Code (Copilot)

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "coffee-grind-analyzer": {
      "type": "stdio",
      "command": "python",
      "args": ["${workspaceFolder}/mcp_server/server.py"]
    }
  }
}
```

## Example usage (from an AI assistant)

> "Analyse this photo of my coffee grounds"

The assistant will:
1. Read the `reference://instructions` resource to understand requirements
2. Call `analyze_grind_photo` with the base64-encoded photo
3. Return the statistics and brew recommendation in natural language
