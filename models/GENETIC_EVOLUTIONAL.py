import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from deap import base, creator, tools, algorithms
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketPredictionSystem:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.features = None
        self.model = None
        self.ga = None
        self.scaler = StandardScaler()

    def load_data(self, filepath):
        try:
            self.data = pd.read_csv(filepath)
            logging.info(f"Data loaded successfully from {filepath}")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def calculate_features(self):
        try:
            self.features = pd.DataFrame()
            for indicator, params in self.config['indicators'].items():
                if hasattr(talib, indicator):
                    self.features[indicator] = getattr(talib, indicator)(self.data['Close'], **params)
                else:
                    logging.warning(f"Indicator {indicator} not found in TALib")
            
            self.features.dropna(inplace=True)
            self.data = self.data.iloc[self.features.index]
            self.features = pd.DataFrame(self.scaler.fit_transform(self.features), columns=self.features.columns)
            logging.info("Features calculated and normalized successfully")
        except Exception as e:
            logging.error(f"Error calculating features: {str(e)}")
            raise

    def create_model(self, architecture):
        model = Sequential()
        model.add(Dense(architecture[0], activation='relu', input_shape=(self.features.shape[1],)))
        model.add(BatchNormalization())
        
        for neurons in architecture[1:-1]:
            model.add(Dense(neurons, activation='relu'))
            model.add(BatchNormalization())
        
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=self.config['learning_rate']),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def calculate_roi(self, predictions, actual_returns):
        positions = np.where(predictions > 0.5, 1, -1)
        roi = np.sum(positions * actual_returns) / len(predictions)
        roi -= self.config['transaction_cost'] * np.sum(np.abs(np.diff(positions)))
        return roi

    def fitness_function(self, individual):
        try:
            selected_features = [self.features.columns[i] for i, gene in enumerate(individual[:len(self.features.columns)]) if gene]
            architecture = [2**i for i, gene in enumerate(individual[len(self.features.columns):]) if gene]
            
            if not selected_features or not architecture:
                return -np.inf, -np.inf

            X = self.features[selected_features]
            y = (self.data['Close'].pct_change().shift(-1) > 0).astype(int)[:-1]
            X = X[:-1]  # Align with target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = self.create_model(architecture)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

            predictions = model.predict(X_test).flatten()
            accuracy = np.mean(np.round(predictions) == y_test)
            
            actual_returns = self.data['Close'].pct_change().iloc[-len(y_test):].values
            roi = self.calculate_roi(predictions, actual_returns)

            return accuracy, roi
        except Exception as e:
            logging.error(f"Error in fitness function: {str(e)}")
            return -np.inf, -np.inf

    def setup_genetic_algorithm(self):
        try:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
            creator.create("Individual", list, fitness=creator.FitnessMulti)

            toolbox = base.Toolbox()
            
            # Gene structure: [feature_selection] + [architecture]
            n_features = len(self.features.columns)
            n_layers = self.config['max_layers']
            
            toolbox.register("attr_bool", random.randint, 0, 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             toolbox.attr_bool, n=n_features + n_layers)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", self.fitness_function)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
            toolbox.register("select", tools.selNSGA2)

            self.ga = {"toolbox": toolbox}
            logging.info("Genetic algorithm setup completed")
        except Exception as e:
            logging.error(f"Error setting up genetic algorithm: {str(e)}")
            raise

    def run_genetic_algorithm(self):
        try:
            pop = self.ga["toolbox"].population(n=self.config['population_size'])
            hof = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)

            pop, log = algorithms.eaMulti(pop, self.ga["toolbox"], 
                                          cxpb=0.7, mutpb=0.2, 
                                          ngen=self.config['generations'], 
                                          stats=stats, halloffame=hof, verbose=True)
            
            best_individual = hof[0]
            logging.info(f"Best individual: {best_individual}")
            logging.info(f"Best fitness: {best_individual.fitness.values}")
            
            return best_individual
        except Exception as e:
            logging.error(f"Error running genetic algorithm: {str(e)}")
            raise

    def create_model_from_individual(self, individual):
        try:
            selected_features = [self.features.columns[i] for i, gene in enumerate(individual[:len(self.features.columns)]) if gene]
            architecture = [2**i for i, gene in enumerate(individual[len(self.features.columns):]) if gene]
            
            X = self.features[selected_features]
            y = (self.data['Close'].pct_change().shift(-1) > 0).astype(int)[:-1]
            X = X[:-1]  # Align with target

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = self.create_model(architecture)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
            
            model.fit(X_train, y_train,