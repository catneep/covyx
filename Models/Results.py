from dataclasses import asdict, dataclass


@dataclass(frozen=False, order=False)
class Result:
    total: float
    score: int
    corads: int
    sections: dict
    drawable: dict

    def _calculate_damage_score(self, damage) -> int:
        """
        Returns a score from 0 - 5 according to the provided damage percentage
        """
        if damage < 17.9:
            return 0

        points = 0
        score_table = (1, 5, 25, 50, 75, 100)
        try:
            while damage > score_table[points]:
                points += 1
        except IndexError:
            if damage > 100:
                return 5
            return 0
        return points

    def _get_corads(self, value: int) -> int:
        """
        Returns a CORADS score for a value
        """
        corads = (0, 21, 21, 21, 25, 100)
        for i, limit in enumerate(corads):
            if value < limit:
                return i

    def _get_damage_scores(self, ratios, frames) -> list:
        result = []
        result.append(ratios[0])
        result.append(ratios[2])
        result.append(ratios[4])

        new_area = sum(frames[:2])
        try:
            percentA = round(frames[0] / new_area, 5)
            percentB = round(frames[1] / new_area, 5)
            result.append(
                round((ratios[1] * percentA) + (ratios[3] * percentB), 5)
            )
        except ZeroDivisionError:
            result.append(0)

        result.append(ratios[5])
        return result

    def __init__(
        self,
        result: float,
        sections: list,
        distribution: list,
        frames: list,
        ratios: list,
    ) -> None:
        damages = self._get_damage_scores(ratios, frames)
        scores = []
        for section in damages:
            score = self._calculate_damage_score(section)
            scores.append(score)

        self.total = round(result, 5)
        self.score = sum(scores)
        self.corads = self._get_corads(self.total)
        self.sections = {"damage": damages, "scores": scores}
        self.drawable = {"damage": sections, "distribution": distribution}

    def __str__(self) -> str:
        return str(asdict(self))
