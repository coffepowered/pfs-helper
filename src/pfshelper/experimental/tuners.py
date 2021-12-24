from ray import tune
from sklearn.metrics import mean_squared_error as mse
from operator import itemgetter
import lightgbm as lgb
import logging

#logger = logging.getLogger(__name__)

# TODO: put a logger
class LightGbmRegressorTrainable(tune.Trainable):
    # Original code from:
    # https://docs.ray.io/en/latest/tune/api_docs/trainable.html#function-api
    
    def setup(self, config:dict, data:dict=None):
        # setup function is invoked once training starts.
        # config (dict): A dict of hyperparameters
        self.x = 0
        
        self.total_loss = 0 
        self.average_loss = 0
        self.losses = {"train": [], "cv": []}
        
        # Hyperparams to tune
        self.lr = config["lr"]
        self.max_depth = config.get("max_depth", None)
        self.n_estimators = config.get("n_estimators", None)
        self.subsample = config.get("subsample", .75)
        self.min_data_in_leaf  = int(config.get("min_data_in_leaf", 20))
        self.max_bin = int(config.get("max_bin", 255))
        
        self.X = data["X"]
        self.y = data["y"]
        self.cv = data["cv"] #list of tuples
        
    def step(self):  # This is called iteratively.
        """
           step is invoked multiple times. 
           Each time, the Trainable object executes one logical iteration of training in the tuning process,
           which may include one or more iterations of actual training.
        """
        
        train_score = None
        cv_score = None
        if self.x <= len(self.cv):
            ti, te = self.cv[self.x]
            
            # Instantiates a new model at each fold
            # https://lightgbm.readthedocs.io/en/latest/Parameters.html?highlight=min_child_samples#learning-control-parameters
            model = lgb.LGBMRegressor(**{"learning_rate": self.lr,
                                         "max_depth": self.max_depth,
                                         "n_estimators": self.n_estimators,
                                         "subsample": self.subsample,
                                         "min_data_in_leaf ": self.min_data_in_leaf,
                                         "max_bin": self.max_bin})
            # fits one fold
            X_train, y_train, X_cv, y_cv = self.X.loc[ti], self.y.loc[ti], self.X.loc[te], self.y.loc[te]
            model.fit(X_train, y_train)
            
            y_hat_train = model.predict(X_train)
            y_hat_cv = model.predict(X_cv)
            
            train_score = mse(y_train, y_hat_train)**.5
            cv_score = mse(y_cv, y_hat_cv)**.5
            
            self.losses["train"].append(train_score)
            self.losses["cv"].append(cv_score)
            
            self.total_loss = self.total_loss + cv_score
            self.average_loss = self.total_loss / (1+self.x)
        else:
            print("Do noting")
        self.x += 1
        return {"score": self.average_loss, "cv_score": cv_score, "train_score": train_score, "losses": self.losses}