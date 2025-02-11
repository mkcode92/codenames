from __future__ import annotations

import random
from pathlib import Path
from typing import Self

from pydantic import BaseModel

with (Path(__file__).parent / "data" / "words-en.lst").open() as src:
    words = sorted(set([line.strip() for line in src]))


class Game(BaseModel):
    # words for the red team
    red: set[str]
    # words for the blue team
    blue: set[str]
    # neutral words
    neutral: set[str]
    # assassin (1)
    assassin: str

    @property
    def all_words(self) -> list[str]:
        return [*self.red, *self.blue, *self.neutral, self.assassin]

    @property
    def categories(self) -> list[str]:
        return (
            ["red"] * len(self.red)
            + ["blue"] * len(self.blue)
            + ["neutral"] * len(self.neutral)
            + ["assassin"]
        )

    def teams(self, red: bool = True) -> list[str]:
        team, other_team = ("red", "blue") if red else ("blue", "red")
        replace = {team: "team", other_team: "other_team"}
        return [replace.get(cat, cat) for cat in self.categories]

    @classmethod
    def create_random(cls, red: bool = True) -> Self:
        n_red, n_blue = (9, 8) if red else (8, 9)
        game_words = random.sample(words, 25)
        return cls(
            red=game_words[:n_red],
            blue=game_words[n_red : n_red + n_blue],
            neutral=game_words[-8:-1],
            assassin=game_words[-1],
        )


class Hint(BaseModel):
    word: str
    score: float
    targets: list[str]


class TeamWeights(BaseModel):
    team: float
    other_team: float
    neutral: float
    assassin: float
