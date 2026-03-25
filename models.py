from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class MLPModel(BaseModel):
    def __init__(
        self,
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=5000,
        alpha=1e-4,
        learning_rate='adaptive',
        learning_rate_init=1e-3,
        validation_fraction=0.1
    ):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=max_iter,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            validation_fraction=validation_fraction
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
