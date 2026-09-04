#!/usr/bin/env bash
# ──────────────────────────────────────────────────────
# SmartAnnotate-AI — One-Click Environment Setup
# ──────────────────────────────────────────────────────
# Usage: bash setup_env.sh
# Supports: Linux, macOS, Windows (Git Bash / WSL)
# ──────────────────────────────────────────────────────

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════╗"
echo "║     🏷️  SmartAnnotate-AI — Environment Setup     ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── 1. Check Python version ──────────────────────
echo -e "${YELLOW}[1/5]${NC} Checking Python installation..."

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON_CMD="$cmd"
            echo -e "  ${GREEN}✓ Found $cmd ($("$cmd" --version 2>&1))${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "  ${RED}✗ Python 3.9+ is required but not found.${NC}"
    echo "  Please install Python 3.9 or later and try again."
    exit 1
fi

# ── 2. Create virtual environment ────────────────
echo -e "${YELLOW}[2/5]${NC} Creating virtual environment..."

VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "  ${YELLOW}⚠ Virtual environment already exists. Reusing.${NC}"
else
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo -e "  ${GREEN}✓ Created virtual environment at .venv/${NC}"
fi

# ── 3. Activate & install dependencies ───────────
echo -e "${YELLOW}[3/5]${NC} Installing dependencies..."

# Determine activation script path
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
else
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
fi

# shellcheck disable=SC1090
source "$ACTIVATE_SCRIPT"

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo -e "  ${GREEN}✓ All dependencies installed${NC}"

# ── 4. Setup environment file ────────────────────
echo -e "${YELLOW}[4/5]${NC} Setting up environment configuration..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "  ${GREEN}✓ Created .env from .env.example${NC}"
    echo -e "  ${YELLOW}⚠ Remember to add your OPENROUTER_API_KEY to .env${NC}"
else
    echo -e "  ${GREEN}✓ .env already exists — skipping${NC}"
fi

# ── 5. Create data directories ───────────────────
echo -e "${YELLOW}[5/5]${NC} Creating data directories..."

mkdir -p data/exports
echo -e "  ${GREEN}✓ data/ and data/exports/ directories ready${NC}"

# ── Done ─────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        ✅ Setup Complete!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${CYAN}To activate the environment:${NC}"
echo -e "    source ${ACTIVATE_SCRIPT}"
echo ""
echo -e "  ${CYAN}To run the application:${NC}"
echo -e "    streamlit run app.py"
echo ""
echo -e "  ${CYAN}To test the pipeline (CLI):${NC}"
echo -e "    python pipeline.py"
echo ""
