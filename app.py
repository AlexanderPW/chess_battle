import logging
import sys
from dotenv import load_dotenv
from arena_chess.ui import make_display

if __name__ == "__main__":
    load_dotenv(override=True)
    
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
    app.launch()
    