from model.classifier import ClassifierBuilder
from model.fair_kernel import FairKernelBuilder
from model.fair_rep_cls import FairRepClsBuilder

from model.sipm import sIPMModelBuilder
from model.laftr import LaftrNetBuilder
from model.cfair import CFairNetBuilder


class ModelFactory(object):
    """Factory class to build new model objects
    """

    def __init__(self):
        self._builders = dict()

    def register_builder(self, key, builder):
        """Registers a new model builder into the factory
        Args:
            key (str): string key of the model builder
            builder (any): Builder object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): string key of the model builder
            **kwargs: keyword arguments
        Returns:
            any: Returns an instance of a model object correspponding to the model builder
        Raises:
            ValueError: If model builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


factory = ModelFactory()
factory.register_builder("fair_rep_cls", FairRepClsBuilder())
factory.register_builder("classifier", ClassifierBuilder())
factory.register_builder("fkernel", FairKernelBuilder())


# other models
factory.register_builder("sipm", sIPMModelBuilder())
factory.register_builder("laftr", LaftrNetBuilder())
factory.register_builder("cfair", CFairNetBuilder())
