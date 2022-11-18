import argparse
import yaml

from trainer import *


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['n_fft'] = config['sample_rate'] * 64 // 1000 + 4
    config['hop_length'] = config['sample_rate'] * 16 // 1000 + 4
    return config


def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run DCUNet.")

    parser.add_argument("--train_config_path",
                        type=str,
                        default='./config/train_config.yaml',
                        help="Config path of trainer.")

    return parser.parse_args()


def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    train_config = load_config(args.train_config_path)
    trainer = DCUNetTrainer(train_config)
    if train_config['load_path']:
        trainer.load()
    else:
        trainer.train()
    # trainer.pesq_score() # 存在问题
    if train_config['save_path']:
        trainer.save()


if __name__ == "__main__":
    main()