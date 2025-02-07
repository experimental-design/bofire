from bofire.strategies.doe_strategy import DoEStrategy
from bofire.strategies.factorial import FactorialStrategy
from bofire.strategies.fractional_factorial import FractionalFactorialStrategy
from bofire.strategies.mapper import map
from bofire.strategies.predictives.active_learning import ActiveLearningStrategy
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.strategies.predictives.enting import EntingStrategy
from bofire.strategies.predictives.multi_fidelity import MultiFidelityStrategy
from bofire.strategies.predictives.predictive import PredictiveStrategy
from bofire.strategies.predictives.qehvi import QehviStrategy
from bofire.strategies.predictives.qnehvi import QnehviStrategy
from bofire.strategies.predictives.qparego import QparegoStrategy
from bofire.strategies.predictives.sobo import (
    AdditiveSoboStrategy,
    CustomSoboStrategy,
    MultiplicativeSoboStrategy,
    SoboStrategy,
)
from bofire.strategies.random import RandomStrategy
from bofire.strategies.shortest_path import ShortestPathStrategy
from bofire.strategies.stepwise.stepwise import StepwiseStrategy
from bofire.strategies.strategy import Strategy
