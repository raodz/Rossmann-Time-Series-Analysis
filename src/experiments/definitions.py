from src.utils import OBVIOUS_FEATURES


def get_experiment_definitions():
    """Return experiment definitions to execute."""
    obvious_features = OBVIOUS_FEATURES

    return [
        # Experiment name, features to exclude, whether to exclude Sundays
        ("baseline", None, False),
        ("no_obvious_features", obvious_features, False),
        ("no_sundays", None, True),
        ("no_obvious_features_no_sundays", obvious_features, True)
    ]