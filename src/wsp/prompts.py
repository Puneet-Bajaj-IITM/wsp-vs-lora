"""The prompt format shared by every task: a system message and a user message ending in lettered options."""
from wsp import config

OPTIONS_HEADER = "Options:"


def render_user_message(body: str, options: list[str], instruction: str) -> str:
    lines = [body, "", OPTIONS_HEADER]
    lines += [f"{letter}. {option}" for letter, option in zip(config.OPTION_LETTERS, options)]
    lines += ["", instruction]
    return "\n".join(lines)


def build_chat(user_message: str, system: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
