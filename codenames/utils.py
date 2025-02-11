from colorama import Fore, Back

from codenames.model import Game

cmap = {
    "red": Fore.RED,
    "blue": Fore.BLUE,
    "neutral": Fore.GREEN,
    "assassin": Fore.BLACK,
}


def print_game(game: Game, sort=True, pad=10, targets: set[str] | None = None):
    words, categories = game.all_words, game.categories
    targets = targets or set()
    if sort:
        words, categories = zip(*sorted(zip(words, categories)))
    for i in range(5):
        cells = []
        line_words = words[i * 5 : (i + 1) * 5]
        line_categories = categories[i * 5 : (i + 1) * 5]
        for word, cat in zip(line_words, line_categories):
            back = Back.YELLOW if word in targets else Back.RESET
            color = cmap[cat]
            ws = " " * (pad - len(word))
            cells.append("".join([back, color, word, ws]))
        print("".join(cells))
    print(Fore.RESET + Back.RESET, end="")
