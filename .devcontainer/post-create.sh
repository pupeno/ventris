#!/usr/bin/env bash

# Fail fast, fail early, fail loud.
set -euo pipefail

workspace_dir="$PWD"

echo "==> Upgrading packages"
sudo apt-get update
sudo apt-get upgrade --yes

echo "==> Installing ripgrep"
sudo apt-get install --yes ripgrep

echo "==> Installing Starship"
sudo apt-get install starship --yes
grep -qxF 'eval "$(starship init bash)"' "$HOME/.bashrc" || echo 'eval "$(starship init bash)"' >> "$HOME/.bashrc"
grep -qxF 'eval "$(starship init zsh)"' "$HOME/.zshrc" || echo 'eval "$(starship init zsh)"' >> "$HOME/.zshrc"
mkdir -p "$HOME/.config"
if [ ! -f "$HOME/.config/starship.toml" ]; then
    cp "$workspace_dir/.devcontainer/starship.toml" "$HOME/.config/starship.toml"
fi

echo "==> Installing Codex and Claude Code"
npm config set allow-scripts=@anthropic-ai/claude-code --location=user
npm install -g @openai/codex @anthropic-ai/claude-code
