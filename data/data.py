from data.adult import AdultBuilder
from data.adult import AdultCondBuilder
from data.adult_cls import AdultClsBuilder

from data.heritage_health import HeritageHealthDatasetBuilder
from data.heritage_health_cls import HeritageHealthClsBuilder

from data.waterbirds import WaterbirdsBuilder
from data.waterbirds_cls import WaterbirdsClsBuilder

from data.compas import CompasBuilder
from data.compas_cls import CompasClsBuilder

class DataFactory(object):
    """Factory class to build new dataset objects
    """

    def __init__(self):
        self._builders = dict()

    def register_builder(self, key, builder):
        """Registers a new dataset builder into the factory
        Args:
            key (str): string key of the dataset builder
            builder (any): Builder object
        """
        self._builders[key] = builder

    def create(self, key, **kwargs):
        """Instantiates a new builder object, once it's registered
        Args:
            key (str): string key of the dataset builder
            **kwargs: keyword arguments
        Returns:
            any: Returns an instance of a dataset object correspponding to the dataset builder
        Raises:
            ValueError: If dataset builder is not registered, raises an exception
        """
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


factory = DataFactory()

factory.register_builder("adult", AdultBuilder())
factory.register_builder("adult-cond", AdultCondBuilder())
factory.register_builder("adult-cls", AdultClsBuilder())

factory.register_builder("heritage", HeritageHealthDatasetBuilder())
factory.register_builder("heritage-cls", HeritageHealthClsBuilder())

factory.register_builder("compas", CompasBuilder())
factory.register_builder("compas-cls", CompasClsBuilder())

factory.register_builder("waterbirds", WaterbirdsBuilder())
factory.register_builder("waterbirds-cls", WaterbirdsClsBuilder())
