from .car_dense_model import CarDenseModel
from .car_simple_model import CarSimpleModel
from .car_model import CarModel


_registry = {
    "car_dense_model": CarDenseModel,
    "car_simple_model": CarSimpleModel,
    "car_model": CarModel
}


def create_model(model_type):

    return _registry[model_type]()
