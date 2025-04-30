import argparse
from pprint import pprint
from asbi.experiments.run import Runner
from asbi.experiments.utils import load_config 

def main():
    # set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    
    # setup paths
    #config_path = f"../configs/{args.config}.yaml"

    # load configuration
    print('loading config...')
    print('config path: ', args.config_path)
    config = load_config(args.config_path)

    print(f'config:')
    pprint(config)
    
    # initialize and run the program
    print('initializing runner...')
    program = Runner(config, args.output_path)

    print('running...')
    program.run()

if __name__ == '__main__':
    main()
