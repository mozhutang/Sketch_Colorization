import argparse

def GetArgs():
    parser = argparse.ArgumentParser(description='Automatically color the sketch based on reference style')
    parser.add_argument(
        '--batchsize',
        help='set number of batch size',
        metavar='',
        type=int,
        default=4)
    parser.add_argument(
        '--epochsnum',
        help='set number of total epochs',
        metavar='',
        type=int,
        default=20)
    parser.add_argument(
        '--learningrate',
        help='set training learning rate',
        metavar='',
        type=float,
        default=0.0002)
    parser.add_argument(
        '--train',
        help='set whether run in train mode or not',
        action='store_true')
    parser.add_argument(
        '--sample',
        help='set number of sample images in validation',
        metavar='',
        type=int,
        default=3)
    return parser