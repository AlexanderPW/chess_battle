---
title: Alex's LLM Chess Battle
emoji: ♟️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.8.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Alex's LLM Chess Battleground

Large Language Models · Battle of Wits

Watch cutting-edge AI models duke it out on the 64 squares. Who reigns supreme?

## 🎮 Features

- **Multi-Model Arena** — Pit any LLM against any other. Claude vs GPT-4? Deepseek vs Llama? You decide.
- **Real-Time Web Interface** — Beautiful Gradio UI displays the board, moves, threats, and player analysis in real time.
- **LLM Reasoning Display** — See exactly what each player is thinking: their evaluation, tactics, plan, and move justification.
- **Intelligent Endgame Engine** — Stockfish automatically takes over in endgames (when either player drops below 3 major pieces) and locks in to deliver decisive, no-draw results.
- **Fair Endgame Assignments** — Stockfish is assigned to the side with more material at endgame entry and stays loyal—no flip-flopping, no nonsense.
- **Smart Move Fallbacks** — LLMs try to play their chosen move. If it fails, the system salvages a legal move from their text. Only then does heuristics kick in.
- **Comprehensive Logging** — Full move-by-move logs show phase, move source (LLM/Stockfish/salvaged/heuristic), and draw reasons.
- **Draw Prevention** — Games end decisively. No 50-move-rule loops, no threefold repetition stalling. Stockfish crushes endgames to checkmate.
- **ELO Leaderboard** — Track model performance with real-time ELO ratings. See which LLM is strongest across all games.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up MongoDB (required for leaderboard):**
   ```bash
   # Run MongoDB in Docker (one-time setup)
   docker run -d \
     --name mongodb_shared \
     -p 27017:27017 \
     -e MONGO_INITDB_ROOT_USERNAME=admin \
     -e MONGO_INITDB_ROOT_PASSWORD=password \
     -v ~/docker_volumes/mongo_data:/data/db \
     mongo:latest
   ```

3. **Create `.env` file with:**
   ```
   MONGO_URI=mongodb://admin:password@localhost:27017/
   OPENAI_API_KEY=sk-proj-...
   ANTHROPIC_API_KEY=sk-ant-...
   DEEPSEEK_API_KEY=sk...
   GROQ_API_KEY=...
   ```

4. **Start the app:**
   ```bash
   python app.py
   ```

5. **Open your browser** to `http://localhost:7860` and watch the magic happen.

## ⚙️ Configuration

Edit `app.py` to set:
- **White model** (Claude, GPT, Ollama/llama, Deepseek, etc.)
- **Black model** (any LLM)
- **Stockfish time limit** (currently 1.0s for strong analysis)

## 📊 Game Flow

1. **Opening/Middlegame** — Both players play pure LLM chess with full reasoning.
2. **Endgame Trigger** — When either player drops below 3 major pieces, Stockfish is assigned to whoever has the advantage.
3. **Endgame Execution** — Assigned player switches to Stockfish-only moves (analyzed by LLM for display). Opponent plays LLM-only chess.
4. **Decisive Win** — Stockfish's raw strength forces checkmate instead of endless draws.

## 📜 Logs

Run with:
```bash
python app.py 2>&1 | tee game.log
```

Logs show:
- Game phase (opening/middlegame/endgame)
- Move source and move played
- When Stockfish is assigned and to whom
- Draw conditions if the game ends in a draw

## 🏗️ Architecture

- **`arena_chess/player.py`** — LLM decision logic, move generation, Stockfish integration
- **`arena_chess/board.py`** — Chess logic wrapper, endgame detection, Stockfish assignment
- **`arena_chess/llm.py`** — Multi-model LLM abstraction layer
- **`arena_chess/game.py`** — Game orchestration
- **`app.py`** — Gradio web UI and application entry point

## 🎯 Why This Works

- **LLM strength** in strategy and piece coordination (opening/middlegame)
- **Stockfish strength** in tactical endgames (forced checkmates)
- **No compromises** — assigned players don't hedge between two engines, they commit to one
- **Transparent reasoning** — see the LLM's thought process even when Stockfish is driving the moves

## 🔧 Optional - Using Ollama

Run local models with Ollama:
1. Download and install from https://ollama.com
2. Run `ollama run llama3.2` (or `llama3.2:1b` for smaller machines)
3. The app will auto-connect to your local Ollama instance

## 📝 Setting up API Keys

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk...
GROQ_API_KEY=...
```

---

Enjoy the battle! ♟️ May the best model win.




