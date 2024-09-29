"""Algorithm registry."""
from simulator.algorithms.predictors.local_prediction import LOC_PREDICTION
from simulator.algorithms.predictors.global_prediction import GLO_PREDICTION

Prediction_REGISTRY = {
    "local_prediction": LOC_PREDICTION,
    "global_prediction": GLO_PREDICTION,
}
