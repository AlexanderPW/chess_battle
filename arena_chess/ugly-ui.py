import gradio as gr
import pandas as pd

from arena.llm import LLM
from .game import Game
from .board import WHITE, BLACK

css = """
.dataframe-fix .table-wrap { min-height: 800px; max-height: 800px; }
footer{display:none !important}
"""

ALL_MODEL_NAMES = LLM.all_model_names()


def message_html(game) -> str:
    return f"<h3>{game.board.message()}</h3>"


def format_records_for_table(games):
    df = pd.DataFrame(
        [
            [g.when, g.red_player, g.yellow_player,
             "White" if g.red_won else "Black" if g.yellow_won else "Draw"]
            for g in reversed(games)
        ],
        columns=["When", "White Player", "Black Player", "Winner"],
    )
    df["When"] = pd.to_datetime(df["When"]).dt.floor("s")
    return df


def format_ratings_for_table(ratings_map):
    items = sorted(ratings_map.items(), key=lambda x: x[1], reverse=True)
    return [[name, int(round(rating))] for name, rating in items]


def load_callback(white_llm, black_llm):
    game = Game(white_llm, black_llm)
    enabled = gr.Button(interactive=True)
    return (
        game,
        game.board.svg(),
        message_html(game),
        "",
        "",
        enabled,
        enabled,
        enabled,
    )


def leaderboard_callback(game):
    records_df = format_records_for_table(Game.get_games())
    ratings_df = format_ratings_for_table(Game.get_ratings())
    return records_df, ratings_df


def move_callback(game):
    game.move()
    if_active = gr.Button(interactive=game.is_active())
    return (
        game,
        game.board.svg(),
        message_html(game),
        game.thoughts(WHITE),
        game.thoughts(BLACK),
        if_active,
        if_active,
    )


def run_callback(game):
    enabled = gr.Button(interactive=True)
    disabled = gr.Button(interactive=False)

    game.reset()
    yield (
        game,
        game.board.svg(),
        message_html(game),
        game.thoughts(WHITE),
        game.thoughts(BLACK),
        disabled, disabled, disabled,
    )

    while game.is_active():
        game.move()
        yield (
            game,
            game.board.svg(),
            message_html(game),
            game.thoughts(WHITE),
            game.thoughts(BLACK),
            disabled, disabled, disabled,
        )

    game.record()
    yield (
        game,
        game.board.svg(),
        message_html(game),
        game.thoughts(WHITE),
        game.thoughts(BLACK),
        disabled, disabled, enabled,
    )


def model_callback(side, game, new_model_name):
    player = game.players[side]
    player.switch_model(new_model_name)
    return game


def white_model_callback(game, new_model_name):
    return model_callback(WHITE, game, new_model_name)


def black_model_callback(game, new_model_name):
    return model_callback(BLACK, game, new_model_name)


def player_section(name, default):
    with gr.Row():
        gr.HTML(f"<h3>{name} Player</h3>")
    with gr.Row():
        dropdown = gr.Dropdown(ALL_MODEL_NAMES, value=default, label="LLM", interactive=True)
    with gr.Row():
        gr.HTML("<h4>Inner thoughts</h4>")
    with gr.Row():
        thoughts = gr.HTML()
    return thoughts, dropdown


def make_display():
    with gr.Blocks(
        title="Alex's LLM Chess Battle",
        css=css,
        theme=gr.themes.Default(),
    ) as blocks:
        game = gr.State()

        with gr.Tabs():
            with gr.TabItem("Game"):
                with gr.Row():
                    gr.HTML("<h2>Alex's Chess LLM Showdown</h2>")

                with gr.Row():
                    with gr.Column(scale=1):
                        white_thoughts, white_dropdown = player_section("White", ALL_MODEL_NAMES[0])

                    with gr.Column(scale=2):
                        message = gr.HTML("<h3>Board</h3>")
                        board_display = gr.HTML()

                        with gr.Row():
                            move_button = gr.Button("Next move")
                            run_button = gr.Button("Run game", variant="primary")
                            reset_button = gr.Button("Start over", variant="stop")

                    with gr.Column(scale=1):
                        black_thoughts, black_dropdown = player_section(
                            "Black", ALL_MODEL_NAMES[1] if len(ALL_MODEL_NAMES) > 1 else ALL_MODEL_NAMES[0]
                        )

            with gr.TabItem("Leaderboard") as leaderboard_tab:
                with gr.Row():
                    with gr.Column(scale=1):
                        ratings_df = gr.Dataframe(
                            headers=["Player", "ELO"],
                            label="Ratings",
                            col_count=2,
                            row_count=10,
                            max_height=800,
                            elem_classes=["dataframe-fix"],
                        )
                    with gr.Column(scale=2):
                        results_df = gr.Dataframe(
                            headers=["When", "White Player", "Black Player", "Winner"],
                            label="Game History",
                            col_count=4,
                            row_count=10,
                            max_height=800,
                            elem_classes=["dataframe-fix"],
                        )

        blocks.load(
            load_callback,
            inputs=[white_dropdown, black_dropdown],
            outputs=[game, board_display, message, white_thoughts, black_thoughts,
                     move_button, run_button, reset_button],
        )

        move_button.click(
            move_callback,
            inputs=[game],
            outputs=[game, board_display, message, white_thoughts, black_thoughts,
                     move_button, run_button],
        )

        run_button.click(
            run_callback,
            inputs=[game],
            outputs=[game, board_display, message, white_thoughts, black_thoughts,
                     move_button, run_button, reset_button],
        )

        reset_button.click(
            load_callback,
            inputs=[white_dropdown, black_dropdown],
            outputs=[game, board_display, message, white_thoughts, black_thoughts,
                     move_button, run_button, reset_button],
        )

        white_dropdown.change(white_model_callback, inputs=[game, white_dropdown], outputs=[game])
        black_dropdown.change(black_model_callback, inputs=[game, black_dropdown], outputs=[game])

        leaderboard_tab.select(leaderboard_callback, inputs=[game], outputs=[results_df, ratings_df])

    return blocks
