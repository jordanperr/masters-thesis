import metrics
import tensorflow
import numpy

def _get_basic_sequential():
    model = tensorflow.keras.Sequential([tensorflow.keras.layers.Dense(3) for i in range(4)])
    model.build(input_shape=(2,2))
    return model

def test_basic_statistics():
    model = _get_basic_sequential()
    basic_statistics = metrics.basic_statistics(model)
    assert len(basic_statistics["weight_avg"]) == 4