import logging
import sys
import os
import subprocess
from dotenv import load_dotenv
from arena_chess.ui import make_display
from arena_chess.player import STOCKFISH_PATH

def stockfish_self_test():
    """Verify Stockfish binary exists and launches correctly."""
    path = "/usr/games/stockfish"
    print(f"Stockfish exists: {os.path.exists(path)}", file=sys.stderr)
    if os.path.exists(path):
        try:
            out = subprocess.run(
                [path],
                input="uci\nquit\n",
                text=True,
                capture_output=True,
                timeout=5
            )
            print(f"Stockfish launch: OK", file=sys.stderr)
            if "uciok" in out.stdout:
                print(f"✓ Stockfish verified at {path}", file=sys.stderr)
            else:
                print(f"⚠ Stockfish output unexpected: {out.stdout[:200]}", file=sys.stderr)
        except Exception as e:
            print(f"✗ Stockfish launch error: {repr(e)}", file=sys.stderr)
    else:
        print(f"✗ Stockfish binary not found at {path}", file=sys.stderr)

if __name__ == "__main__":
    load_dotenv(override=True)
    
    # Verify Stockfish availability at startup
    if STOCKFISH_PATH:
        print(f"✓ Stockfish found at: {STOCKFISH_PATH}", file=sys.stderr)
        stockfish_self_test()
    else:
        print("⚠ WARNING: Stockfish not found. Endgame moves will use LLM only.", file=sys.stderr)
    
    # Configure logging to stderr with detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
        force=True
    )
    
    # Suppress verbose engine protocol logging
    logging.getLogger('chess.engine').setLevel(logging.WARNING)
    
    app = make_display()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
    