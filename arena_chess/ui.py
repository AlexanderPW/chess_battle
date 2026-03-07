import gradio as gr
import pandas as pd
import chess

from .llm import LLM
from .game import Game
from .board import WHITE, BLACK

ALL_MODEL_NAMES = LLM.all_model_names()

# ── Piece unicode maps ─────────────────────────────────────────────────────────
PIECE_UNICODE = {
    (chess.PAWN,   chess.WHITE): "♙", (chess.KNIGHT, chess.WHITE): "♘",
    (chess.BISHOP, chess.WHITE): "♗", (chess.ROOK,   chess.WHITE): "♖",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.PAWN,   chess.BLACK): "♟", (chess.KNIGHT, chess.BLACK): "♞",
    (chess.BISHOP, chess.BLACK): "♝", (chess.ROOK,   chess.BLACK): "♜",
    (chess.QUEEN,  chess.BLACK): "♛",
}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PIECE_ORDER  = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]


def _captured_and_advantage(board: chess.Board):
    start = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
    white_lost, black_lost = {}, {}
    for pt, n in start.items():
        white_lost[pt] = max(0, n - len(board.pieces(pt, chess.WHITE)))
        black_lost[pt] = max(0, n - len(board.pieces(pt, chess.BLACK)))

    def pieces_str(lost_dict, color):
        parts = [PIECE_UNICODE[(pt, color)] * lost_dict.get(pt, 0)
                 for pt in PIECE_ORDER if lost_dict.get(pt, 0)]
        return "".join(parts) if parts else ""

    white_cap = pieces_str(black_lost, chess.BLACK)
    black_cap = pieces_str(white_lost, chess.WHITE)
    white_val = sum(PIECE_VALUES[pt] * cnt for pt, cnt in black_lost.items())
    black_val = sum(PIECE_VALUES[pt] * cnt for pt, cnt in white_lost.items())
    return white_cap, black_cap, white_val - black_val


def side_panel_html(game, side: str) -> str:
    """Captured pieces + advantage label shown inside each player's column."""
    accent = "#b8860b" if side == "white" else "#c0392b"
    try:
        w_cap, b_cap, adv = _captured_and_advantage(game.board.board)
    except Exception:
        return ""

    cap_str = w_cap if side == "white" else b_cap
    pieces_display = (
        f'<span style="font-size:15px;letter-spacing:2px;">{cap_str}</span>'
        if cap_str else
        '<span style="color:#ddd;font-size:11px;">—</span>'
    )

    adv_label = ""
    if side == "white" and adv > 0:
        adv_label = f'<span style="color:{accent};font-size:11px;font-weight:700;margin-left:6px;">+{adv}</span>'
    elif side == "black" and adv < 0:
        adv_label = f'<span style="color:{accent};font-size:11px;font-weight:700;margin-left:6px;">+{abs(adv)}</span>'

    return f"""
    <div style="padding:6px 0 4px;border-top:1px solid #f0f0f0;margin-top:6px;">
      <span style="font-size:9px;letter-spacing:.1em;text-transform:uppercase;
                   color:#ccc;font-family:sans-serif;">Captured</span>{adv_label}
      <div style="margin-top:3px;min-height:20px;">{pieces_display}</div>
    </div>"""


def message_html(game) -> str:
    msg  = game.board.message()
    font = "font-family:'Cinzel',serif"
    if "wins" in msg.lower():
        icon  = "♚" if "Black" in msg else "♔"
        color = "#c0392b" if "Black" in msg else "#b8860b"
        return f'<div style="text-align:center;{font};font-size:1.1rem;color:{color};padding:6px 0;">{icon} {msg} {icon}</div>'
    if "draw" in msg.lower():
        return f'<div style="text-align:center;{font};font-size:1rem;color:#888;padding:6px 0;">½ — ½ Draw</div>'
    check = "in check" in msg.lower()
    color = "#e05050" if check else "#aaa"
    warn  = "⚠ " if check else ""
    return f'<div style="text-align:center;{font};font-size:.9rem;color:{color};padding:4px 0;">{warn}{msg}</div>'


def thoughts_html(raw: str, side: str) -> str:
    accent = "#b8860b" if side == "white" else "#c0392b"
    label  = "♔ White · Thoughts" if side == "white" else "♚ Black · Thoughts"
    return f"""
    <div style="border:1px solid #f0f0f0;border-radius:6px;padding:10px 12px;
                font-family:'JetBrains Mono',monospace;font-size:11px;line-height:1.6;
                color:#666;max-height:260px;overflow-y:auto;white-space:pre-wrap;">
      <div style="color:{accent};font-size:9px;letter-spacing:.12em;text-transform:uppercase;
                  margin-bottom:6px;font-family:sans-serif;">{label}</div>
      {raw}
    </div>"""


def format_records_for_table(games):
    df = pd.DataFrame(
        [[g.when, g.white_player, g.black_player,
          "White" if g.white_won else "Black" if g.black_won else "Draw"]
         for g in reversed(games)],
        columns=["When", "White Player", "Black Player", "Winner"],
    )
    df["When"] = pd.to_datetime(df["When"]).dt.floor("s")
    return df


def format_ratings_for_table(ratings_map):
    items = sorted(ratings_map.items(), key=lambda x: x[1], reverse=True)
    return [[name, int(round(rating))] for name, rating in items]


# ── Callbacks ─────────────────────────────────────────────────────────────────

def load_callback(white_llm, black_llm):
    game    = Game(white_llm, black_llm)
    enabled = gr.Button(interactive=True)
    return (
        game,
        game.board.svg(),
        message_html(game),
        side_panel_html(game, "white"),
        side_panel_html(game, "black"),
        "", "",
        enabled, enabled, enabled,
    )


def leaderboard_callback(game):
    return (format_records_for_table(Game.get_games()),
            format_ratings_for_table(Game.get_ratings()))


def move_callback(game):
    game.move()
    active = gr.Button(interactive=game.is_active())
    return (
        game,
        game.board.svg(),
        message_html(game),
        side_panel_html(game, "white"),
        side_panel_html(game, "black"),
        game.thoughts(WHITE),
        game.thoughts(BLACK),
        active, active,
    )


def run_callback(game):
    enabled  = gr.Button(interactive=True)
    disabled = gr.Button(interactive=False)
    game.reset()
    yield (game, game.board.svg(), message_html(game),
           side_panel_html(game, "white"), side_panel_html(game, "black"),
           game.thoughts(WHITE), game.thoughts(BLACK),
           disabled, disabled, disabled)

    while game.is_active():
        game.move()
        yield (game, game.board.svg(), message_html(game),
               side_panel_html(game, "white"), side_panel_html(game, "black"),
               game.thoughts(WHITE), game.thoughts(BLACK),
               disabled, disabled, disabled)

    game.record()
    yield (game, game.board.svg(), message_html(game),
           side_panel_html(game, "white"), side_panel_html(game, "black"),
           game.thoughts(WHITE), game.thoughts(BLACK),
           disabled, disabled, enabled)


def white_model_callback(game, name):
    game.players[WHITE].switch_model(name)
    return game

def black_model_callback(game, name):
    game.players[BLACK].switch_model(name)
    return game


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Raleway:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap');

footer, .footer, div[class*="footer"], .built-with, .share-button { display:none !important; }

#board-display { width:100%; overflow:hidden; }
#board-display svg { width:100% !important; height:auto !important; max-width:460px; display:block; margin:0 auto; }

.gr-column, [class*="column"] { min-width:0 !important; }

/* Hide horizontal scrollbars on tables */
table { overflow:hidden !important; }

@media (max-width:860px) {
    .gradio-row { flex-direction:column !important; }
    .gradio-column, .gr-column { width:100% !important; flex:none !important; }
    #board-display svg { max-width:100% !important; }
}
"""

HEADER_HTML = """
<div style="text-align:center;padding:12px 0 8px;border-bottom:1px solid #e8e8e8;margin-bottom:2px;">
  <div style="font-family:'Cinzel',serif;font-size:1.5rem;font-weight:700;
              letter-spacing:.05em;color:#b8860b;line-height:1;">
    ♟ Alex's LLM Chess Battleground
  </div>
  <div style="font-family:'Raleway',sans-serif;font-size:9px;letter-spacing:.16em;
              color:#ccc;text-transform:uppercase;margin-top:4px;">
    Large Language Models · Battle of Wits
  </div>
</div>
"""


def _player_column(label: str, default: str, side: str):
    accent = "#b8860b" if side == "white" else "#c0392b"
    icon   = "♔" if side == "white" else "♚"
    with gr.Column(scale=1):
        gr.HTML(f"""
        <div style="border-bottom:1px solid #e8e8e8;padding-bottom:6px;margin-bottom:8px;">
          <span style="font-family:'Cinzel',serif;font-size:1rem;color:{accent};">
            {icon} {label}
          </span>
        </div>""")
        dropdown  = gr.Dropdown(ALL_MODEL_NAMES, value=default, label="Model", interactive=True)
        cap_panel = gr.HTML()
        thoughts  = gr.HTML(thoughts_html("Waiting for first move…", side))
    return dropdown, cap_panel, thoughts


# ── Layout ────────────────────────────────────────────────────────────────────

def make_display():
    with gr.Blocks(title="Alex's LLM Chess Battleground",
                   css=CSS, theme=gr.themes.Default()) as blocks:
        game = gr.State()

        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            with gr.TabItem("  ♟  Game  "):
                with gr.Row(equal_height=False):

                    white_dropdown, white_cap, white_thoughts = _player_column(
                        "White", ALL_MODEL_NAMES[0], "white"
                    )

                    with gr.Column(scale=2):
                        message    = gr.HTML()
                        board_html = gr.HTML(elem_id="board-display")
                        with gr.Row():
                            move_button  = gr.Button("Next Move",   variant="secondary")
                            run_button   = gr.Button("▶  Run Game", variant="primary")
                            reset_button = gr.Button("↺  Reset",    variant="stop")

                    black_dropdown, black_cap, black_thoughts = _player_column(
                        "Black",
                        ALL_MODEL_NAMES[1] if len(ALL_MODEL_NAMES) > 1 else ALL_MODEL_NAMES[0],
                        "black"
                    )

            with gr.TabItem("  ♛  Leaderboard  ") as leaderboard_tab:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=350):
                        ratings_df = gr.Dataframe(
                            headers=["Player", "ELO"],
                            label="Ratings", col_count=2, row_count=15, max_height=800,
                            wrap=True,
                        )
                    with gr.Column(scale=2, min_width=550):
                        results_df = gr.Dataframe(
                            headers=["When", "White Player", "Black Player", "Winner"],
                            label="Game History", col_count=4, row_count=15, max_height=800,
                            wrap=True,
                        )

        LOAD_OUTPUTS = [game, board_html, message,
                        white_cap, black_cap,
                        white_thoughts, black_thoughts,
                        move_button, run_button, reset_button]

        MOVE_OUTPUTS = [game, board_html, message,
                        white_cap, black_cap,
                        white_thoughts, black_thoughts,
                        move_button, run_button]

        RUN_OUTPUTS  = [game, board_html, message,
                        white_cap, black_cap,
                        white_thoughts, black_thoughts,
                        move_button, run_button, reset_button]

        blocks.load(load_callback,
                    inputs=[white_dropdown, black_dropdown],
                    outputs=LOAD_OUTPUTS)

        move_button.click(move_callback,   inputs=[game], outputs=MOVE_OUTPUTS)
        run_button.click(run_callback,     inputs=[game], outputs=RUN_OUTPUTS)
        reset_button.click(load_callback,
                           inputs=[white_dropdown, black_dropdown],
                           outputs=LOAD_OUTPUTS)

        white_dropdown.change(white_model_callback,
                              inputs=[game, white_dropdown], outputs=[game])
        black_dropdown.change(black_model_callback,
                              inputs=[game, black_dropdown], outputs=[game])

        leaderboard_tab.select(leaderboard_callback,
                               inputs=[game], outputs=[results_df, ratings_df])

    return blocks
