from trainer.classifier import ClassifierBuilder
from trainer.fkernel_dp import FairKernelDPTrainerBuilder
from trainer.frep_dp import FRepDPTrainerBuilder
# from trainer.fair_kernel_eq import FairKernelEqTrainerBuilder
from trainer.frep_eq import FRepEqTrainerBuilder

from trainer.sipm import sIPMTrainerBuilder
from trainer.cfair_laftr import CfairLaftrTrainerBuilder
from trainer.mmd_classifier import MMDClassifierBuilder

# from trainer.dp_9 import DP9TrainerBuilder
# from trainer.eq_9 import Eq9TrainerBuilder

class TrainerFactory(object):

    """Factory class to build new dataset objects
    """

    def __init__(self):
        self._builders = dict()

    def register_builder(self, key, builder):
        """Registers a new trainer builder into the factory
        Args:
            key (str): string key of the trainer builder
            builder (any): Builder object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): string key of the trainer builder
            **kwargs: keyword arguments
        Returns:
            any: Returns an instance of a trainer object correspponding to the trainer builder
        Raises:
            ValueError: If trainer builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


factory = TrainerFactory()
factory.register_builder("classifier", ClassifierBuilder())
factory.register_builder("fkernel_dp", FairKernelDPTrainerBuilder())
factory.register_builder("frep-dp", FRepDPTrainerBuilder())
# factory.register_builder("fair_kernel_eq", FairKernelEqTrainerBuilder())
factory.register_builder("frep-eq", FRepEqTrainerBuilder())

# factory.register_builder("dp9", DP9TrainerBuilder())
# factory.register_builder("eq9", Eq9TrainerBuilder())

# # other methods
factory.register_builder("sipm", sIPMTrainerBuilder())
factory.register_builder("cfair-laftr", CfairLaftrTrainerBuilder())
factory.register_builder("mmd-classifier", MMDClassifierBuilder())
