from solver import Solver
import ConfigParser
import sys


def train_end2end():
    config = ConfigParser.RawConfigParser()
    config_path = sys.argv[1]
    config.read(config_path)

    model = Solver(config)
    model.fit()

if __name__ == '__main__':
    train_end2end()
