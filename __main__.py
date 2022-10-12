import argparse
from dataclasses import asdict
from json import dumps
from time import monotonic

from covyx import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Analyze a CT video,
        see https://github.com/catneep/covyx#readme
        for more information.
      """
    )

    parser.add_argument("path", type=str, help="Path to local .mp4 file")
    parser.add_argument(
        "--time",
        dest="time",
        default=False,
        action="store_true",
        help="output the analysis time",
    )
    parser.add_argument(
        "--pretty",
        dest="pretty",
        default=False,
        action="store_true",
        help="pretty print the result",
    )

    args = parser.parse_args()

    path, time, pretty = args.path, args.time, args.pretty

    start = monotonic()
    result = analyze(path)
    end = monotonic() - start

    result_dict = asdict(result)

    if time:
        result_dict["runtime"] = round(end, 4)

    if pretty:
        print(dumps(result_dict, indent=2, sort_keys=True))
    else:
        print(result_dict)
