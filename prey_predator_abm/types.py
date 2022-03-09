from dataclasses import dataclass
from typing import Annotated, Type, TypedDict
import numpy as np

@dataclass
class PlayerParams():
    proportional_weight: float
    integral_weight: float
    derivative_weight: float
    integral_period: int
    integral_terms: int

    @property
    def total_weight(self):
        return self.proportional_weight + self.integral_weight + self.derivative_weight

    @property
    def alpha(self):
        return self.proportional_weight / self.total_weight

    @property
    def beta(self):
        return self.integral_weight / self.total_weight

    @property
    def gamma(self):
        return self.derivative_weight / self.total_weight

    @property
    def minimum_terms(self):
        return self.integral_period + self.integral_terms + 1


@dataclass
class Player():
    player_no: int
    likelihood_vector: Annotated[dict[int, float], 'player_no -> value']
    params: PlayerParams

    def likelihood(self,
                   actions: list[int]):
        """

        actions: {time_no: action_no}
        """

        if len(actions) > 1:

            def estimator(lst):
                return sum(lst) / len(lst)

            proportional = estimator(actions)

            past_estimators = []
            if len(actions) > (self.params.minimum_terms):
                for i in range(self.params.integral_terms):
                    n = len(actions)
                    eff_actions = actions[n -
                                          (self.params.integral_period + i):(n - i)]
                    past_estimators.append(estimator(eff_actions))
                integral = sum(past_estimators) / len(past_estimators)
                derivative = (past_estimators[0] - past_estimators[1]) / 2
            else:
                integral = proportional
                derivative = proportional

            likelihood = self.params.alpha * proportional
            likelihood += self.params.beta * integral
            likelihood += self.params.gamma * derivative
            return likelihood
        else:
            return 1 / 2


class GameMiningParams(TypedDict):
    shock_timestep: int
    shock_tensor: np.ndarray


# annot: dict[str, type] = GameMiningParams.__annotations__
# sweep_fields: dict[str, type] = {k: list[v] for k, v in annot.items()}
# GameMiningSweepParams = TypedDict('GameMiningSweepParams', sweep_fields)

class GameMiningState(TypedDict):
    timestep: int
    players: list
    actions: dict
    past_actions: dict[int, list]
    payoffs: dict
    payoff_tensor: np.ndarray
