"""
Sample code of how to implement data pipeline with multiple transformers

Pipelines will be part of ColumnTransformers, at each step of ColumnTransformer,
select the columns to be passed into the respective pipelines
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from transformers import NumericalTransformer, CategoricalTransformer


def data_pipe() -> ColumnTransformer:
    """
    Initialize data pipeline

    Returns:
        ColumnTransformer: Column Transformer object
    """

    # ===== SAMPLE CODE =====
    numerical_pipe_1 = Pipeline(
        steps=[
            ("mode_imputer", SimpleImputer(strategy="most_frequent")),
            ("ss", StandardScaler()),
        ]
    )

    numerical_pipe_2 = Pipeline(
        steps=[
            ("mean_imputer", SimpleImputer(strategy="mean")),
            ("ss", StandardScaler()),
        ]
    )

    cat_pipeline_1 = Pipeline(
        steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))]
    )

    cat_pipeline_2 = Pipeline(steps=[("oe", OrdinalEncoder())])

    cat_processing = Pipeline(steps=[("cat_trans", CategoricalTransformer())])
    num_processing = Pipeline(steps=[("num_trans", NumericalTransformer())])

    featurisation = ColumnTransformer(
        transformers=[
            ("cat_processing", cat_processing, ["a", "b", "c", "d", "e", "f"])(
                "ohe", cat_pipeline_1, ["a", "b", "c"]
            ),
            ("oe", cat_pipeline_2, ["d", "e", "f"]),
            ("num_1", numerical_pipe_1, ["g", "h", "i"]),
            ("num_2", numerical_pipe_2, ["j", "k", "l"]),
            ("num_processing", num_processing, ["g", "h", "i", "j", "k", "l"]),
        ],
        remainder="passthrough",
    )

    return featurisation
