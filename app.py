import logging
import sys
from dotenv import load_dotenv
from arena_chess.ui import make_display
from arena_chess.player import STOCKFISH_PATH

if __name__ == "__main__":
    load_dotenv(override=True)
    
    # Verify Stockfish availability at startup
    if STOCKFISH_PATH:
        print(f"✓ Stockfish found at: {STOCKFISH_PATH}", file=sys.stderr)
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
    