import argparse


def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the Bitcoin OTC dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description="Run SGCN.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="./data/labels.txt",
                        help="Edge list csv.")

    parser.add_argument("--features-path",
                        nargs="?",
                        default="./data/Drug-fingerprint.pkl",
                        help="Edge list csv.")

    parser.add_argument("--embedding-path",
                        nargs="?",
                        default="./output/embedding/drug_sgcn.csv",
                        help="Target embedding csv.")

    parser.add_argument("--regression-weights-path",
                        nargs="?",
                        default="./weights/drug_sgcn.csv",
                        help="Regression weights csv.")

    parser.add_argument("--log-path",
                        nargs="?",
                        default="./logs/drug_drug_cmap_logs.json",
                        help="Log json.")

    parser.add_argument("--epochs",
                        type=int,
                        default=1000,
                        help="Number of training epochs. Default is 1000.")

    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30,
                        help="Number of SVD iterations. Default is 30.")

    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=300,
                        help="Number of SVD feature extraction dimensions. Default is 64.")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--lamb",
                        type=float,
                        default=0.1,
                        help="Embedding regularization parameter. Default is 1.0. best now 0.0125, 0.001")

    parser.add_argument("--get_embed",
                        type=bool,
                        default=True,
                        help="Test dataset size. Default is 0.2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=1e-5,
                        help="Learning rate. Default is 10^-5.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help="Layer dimensions separated by space. E.g. 32 32. best now 128")

    parser.add_argument("--spectral-features",
                        default=False)
    parser.add_argument("--general-features",
                        default=True)
    parser.add_argument("--maccskeys",
                          default=False)

    parser.add_argument("--fold",
                        default=5,
                        help="The number of training rounds. Default is 5")

    parser.set_defaults(layers=[64,64])

    return parser.parse_args()
