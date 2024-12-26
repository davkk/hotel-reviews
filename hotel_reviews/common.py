import argparse
import sys
from dataclasses import dataclass
from io import FileIO

import scienceplots
from matplotlib import pyplot as plt


def setup_pyplot():
    plt.style.use(["science", "ieee"])

    # custom font
    plt.rcParams["text.latex.preamble"] += r"""\usepackage[T1]{fontenc}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
%\usepackage[sfdefault,scale=0.95]{FiraSans}
\usepackage[lf]{Baskervaldx} % lining figures
\usepackage[bigdelims,vvarbb]{newtxmath} % math italic letters from nimbus Roman
\usepackage[cal=boondoxo]{mathalfa} % mathcal from STIX, unslanted a bit
\renewcommand*\oldstylenums[1]{\textosf{#1}}"""

    plt.rcParams["path.simplify"] = True

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"

    plt.rcParams["axes.formatter.limits"] = -4, 4
    plt.rcParams["axes.formatter.use_mathtext"] = True

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["*", "1", "+", "2", ".", "3"]
    return colors, markers


@dataclass
class Args:
    infile: FileIO
    outfile: FileIO
    test: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default=sys.stdin,
        type=argparse.FileType("r"),
    )
    parser.add_argument(
        "--outfile",
        default=sys.stdout,
        type=argparse.FileType("w"),
    )
    parser.add_argument("--test", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return Args(**args.__dict__)
