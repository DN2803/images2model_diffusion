import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('DiffMatch', add_help=False)
    parser.add_argument('--train_module', type=str, help='Name of module in the "train_settings/" folder.')
    parser.add_argument('--train_name', type=str, help='Name of the train settings file.')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,
                        help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--seed', type=int, default=1992, help='Pseudo-RNG seed')
    parser.add_argument('--name', type=str, default="Default", help='Name of the experiment')
    parser.add_argument('--corruption', action='store_true')


    return parser