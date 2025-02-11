import abc
from itertools import combinations
from typing import Iterator, Callable, Sequence

import torch
import torch.nn as nn
from scipy.special import comb
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from codenames.model import Game, Hint, TeamWeights
from codenames.utils import print_game

device = "cuda" if torch.cuda.is_available() else "cpu"

MASK_FILLING_TEMPLATE = (
    "The words {} can be thought of together with the word [MASK], "
    + "maybe also through different semantic aspects."
)
MASK_FILLING_WITH_EXCLUSION_TEMPLATE = (
    "The word [MASK] can be thought of together with the words {included}, "
    + "but does not relate to any of the words {excluded}."
)


class HintStrategy:
    @abc.abstractmethod
    def __call__(self, game: Game, hint_for_red=True) -> Iterator[Hint]:
        pass

    def top_k(
        self,
        game: Game,
        k: int = 25,
        min_len: int = 3,
        hint_for_red: bool = True,
        only_best_per_word: bool = True,
    ) -> list[Hint]:
        seen = set()
        hints = []
        for hint in self(game, hint_for_red):
            if len(hint.targets) < min_len:
                continue
            if only_best_per_word:
                if hint.word in seen:
                    continue
                seen.add(hint.word)
            hints.append(hint)
        by_score = sorted(hints, key=lambda hint: hint.score, reverse=True)
        return by_score[:k]

    def print_top_k(self, game: Game, *args, **kwargs) -> None:
        print(
            self.__class__.__name__,
            f"({'red' if kwargs.get('hint_for_red', True) else 'blue'})",
        )
        hints = self.top_k(game, *args, **kwargs)
        for i, hint in enumerate(hints):
            print(f"{i + 1}:", hint.word, f"{hint.score:.03}")
            print_game(game, targets=hint.targets)
            print()


Words2Ids = Callable[[list[str]], list[int]]
SubsetGenerator = Iterator[list[str]]


class EmbeddingSimilarity(HintStrategy):
    default_weights = TeamWeights(team=1, other_team=-5, neutral=-2, assassin=-10)

    def __init__(
        self,
        embeddings: nn.Embedding,
        words2ids: Words2Ids,
        hint_words: dict[str, int],
        weights: TeamWeights | None = None,
    ):
        self.embeddings = embeddings
        self.words2ids = words2ids
        self.hint_words = hint_words
        self.weights = weights or self.default_weights

    def __call__(self, game: Game, hint_for_red=True) -> Iterator[Hint]:
        words = game.all_words
        similarities = self.calculate_similarities(game)

        # incrementally calculate the score
        for hint_word, per_hint_similarities in zip(self.hint_words, similarities):
            score = 0
            targets = []
            # order by high similarity
            for similarity, word, cat in sorted(
                zip(per_hint_similarities, words, game.teams(hint_for_red)),
                reverse=True,
            ):
                targets.append(word)
                # The score is affected by the similarity itself
                # and the weight per category (which also affects the polarity)
                score += similarity * getattr(self.weights, cat)
                yield Hint(word=hint_word, score=score, targets=targets)

    def calculate_similarities(self, game: Game) -> torch.Tensor:
        # calculate all similarities between hint words and game words
        with torch.no_grad():
            game_word_ids = self.words2ids(game.all_words)
            game_embs = self.embeddings(
                torch.tensor(game_word_ids)
            )  # shape: (25, n_emb)
            hint_word_embs = self.embeddings(
                torch.tensor(list(self.hint_words.values()))
            )  # shape: (n_vocab, n_emb)
            similarities = cosine_similarity(
                hint_word_embs, game_embs
            )  # shape (n_vocab, n_emb)
        return similarities


class MaskFilling(HintStrategy):
    default_weights = TeamWeights(team=1, other_team=-3, neutral=-1, assassin=-100)

    def __init__(
        self,
        checkpoint: str = "bert-base-uncased",
        weights: TeamWeights | None = None,
        max_good_targets: int = 4,
        max_bad_targets: int = 1,
        template: str = MASK_FILLING_TEMPLATE,
    ):
        self.unmasker = pipeline("fill-mask", model=checkpoint, device=device)
        self.weights = weights or self.default_weights
        self.max_good_targets = max_good_targets
        self.max_bad_targets = max_bad_targets
        self.template = template

    def __call__(self, game: Game, hint_for_red=True) -> Iterator[Hint]:
        all_words = set(game.all_words)
        teams = game.teams(red=hint_for_red)
        word2score = {
            word: getattr(self.weights, cat) for word, cat in zip(game.all_words, teams)
        }

        team, other_team = (
            (game.red, game.blue) if hint_for_red else (game.blue, game.red)
        )
        subset_generator = lambda: self.candidate_subsets(
            team, other_team, game.neutral
        )
        with torch.no_grad():
            inputs = self.create_inputs(subset_generator, game, hint_for_red)
            all_results = self.unmasker(inputs)
            for results, subset in zip(all_results, subset_generator()):
                subset_score = sum(word2score[word] for word in subset)
                for result in results:
                    # exclude tokens are contained in the game
                    if result["token_str"] in all_words:
                        continue
                    yield Hint(
                        word=result["token_str"],
                        score=result["score"] * subset_score,
                        targets=subset,
                    )

    @staticmethod
    def words2string(words: Sequence[str]) -> str:
        return ", ".join(words[:-1]) + f" and {words[-1]}"

    def create_inputs(
        self,
        subset_generator: Callable[[], SubsetGenerator],
        game: Game,
        hint_for_red: bool,
    ) -> list[str]:
        return [
            self.template.format(self.words2string(subset))
            for subset in subset_generator()
        ]

    def candidate_subsets(
        self, team: set[str], other_team: set[str], neutral: set[str]
    ) -> SubsetGenerator:
        # subsets containing words of team
        def all_good_targets():
            for n_good_targets in range(2, min(self.max_good_targets, len(team)) + 1):
                for combination in combinations(team, n_good_targets):
                    yield combination

        # subsets containing words of other team or neutral
        def all_bad_targets():
            for n_bad_targets in range(self.max_bad_targets):
                for combination in combinations(
                    other_team.union(neutral), n_bad_targets + 1
                ):
                    yield combination

        for good_targets in all_good_targets():
            yield good_targets
            for bad_targets in all_bad_targets():
                yield good_targets + bad_targets

    def n_subsets(self, team: set[str], other_team: set[str], neutral: set[str]) -> int:
        n_good = len(team)
        n_bad = len(other_team.union(neutral))
        good_combos = sum(
            comb(n_good, k, exact=True)
            for k in range(2, min(self.max_good_targets, n_good) + 1)
        )
        bad_combos = sum(
            comb(n_bad, k, exact=True)
            for k in range(1, min(self.max_bad_targets, n_bad) + 1)
        )
        return good_combos + good_combos * bad_combos


class MaskFillingWithExclusion(MaskFilling):
    def create_inputs(
        self,
        subset_generator: Callable[[], SubsetGenerator],
        game: Game,
        hint_for_red: bool,
    ) -> list[str]:
        inputs = []
        bad_terms = set(game.blue if hint_for_red else game.red).union([game.assassin])
        for included in subset_generator():
            excluded = list(bad_terms.difference(included))
            inputs.append(
                self.template.format(
                    included=self.words2string(included),
                    excluded=self.words2string(excluded),
                )
            )
        return inputs
