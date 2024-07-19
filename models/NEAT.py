import neat
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import talib
import multiprocessing
import visualize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialNEAT:
    def __init__(self, config_file, data_file):
        self.config_file = config_file
        self.data_file = data_file
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        try:
            data = pd.read_csv(self.data_file)
            logging.info("Data loaded successfully")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def calculate_features(self, data):
        try:
            features = pd.DataFrame()
            features['SMA'] = talib.SMA(data['Close'], timeperiod=14)
            features['EMA'] = talib.EMA(data['Close'], timeperiod=14)
            features['RSI'] = talib.RSI(data['Close'], timeperiod=14)
            features['MACD'], _, _ = talib.MACD(data['Close'])
            features['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
            features['OBV'] = talib.OBV(data['Close'], data['Volume'])
            
            features.dropna(inplace=True)
            data = data.iloc[features.index]
            
            scaler = StandardScaler()
            features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
            
            logging.info("Features calculated successfully")
            return features, data
        except Exception as e:
            logging.error(f"Error calculating features: {str(e)}")
            raise

    def prepare_data(self):
        data = self.load_data()
        features, data = self.calculate_features(data)
        
        self.X = features.values
        self.y = (data['Close'].pct_change().shift(-1) > 0).astype(int).values[:-1]
        self.X = self.X[:-1]  # Align with target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        
        logging.info("Data prepared successfully")

    def eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        predictions = []
        for xi in self.X_train:
            output = net.activate(xi)
            predictions.append(output[0])
        
        predictions = np.array(predictions)
        accuracy = np.mean((predictions > 0.5) == self.y_train)
        
        # Calculate ROI
        positions = np.where(predictions > 0.5, 1, -1)
        returns = self.y_train * positions
        roi = np.mean(returns)
        
        return accuracy, roi

    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            accuracy, roi = self.eval_genome(genome, config)
            genome.fitness = accuracy * 0.5 + roi * 0.5  # Equal weight to accuracy and ROI

    def run(self, n_generations):
        # Load configuration
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             self.config_file)

        # Create the population
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to n_generations
        winner = p.run(self.eval_genomes, n_generations)

        # Display the winning genome
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, yi in zip(self.X_test, self.y_test):
            output = winner_net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, yi, output))

        # Visualize the network
        node_names = {-1: 'SMA', -2: 'EMA', -3: 'RSI', -4: 'MACD', -5: 'ATR', -6: 'OBV', 0: 'Buy/Sell'}
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)

        return winner, stats

if __name__ == '__main__':
    financial_neat = FinancialNEAT('neat_config.txt', 'financial_data.csv')
    financial_neat.prepare_data()
    winner, stats = financial_neat.run(100)  # Run for 100 generations

    # Use the winner to make predictions on test data
    winner_net = neat.nn.FeedForwardNetwork.create(winner, neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                                                       neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                                                       'neat_config.txt'))
    test_predictions = [winner_net.activate(xi)[0] for xi in financial_neat.X_test]
    test_accuracy = np.mean((np.array(test_predictions) > 0.5) == financial_neat.y_test)
    print(f"Test Accuracy: {test_accuracy}")

    # Calculate ROI on test data
    test_positions = np.where(np.array(test_predictions) > 0.5, 1, -1)
    test_returns = financial_neat.y_test * test_positions
    test_roi = np.mean(test_returns)
    print(f"Test ROI: {test_roi}")