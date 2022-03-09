from game_mining_demo import default_run_args
from cadCAD_tools.execution import easy_run
from datetime import datetime
import click
import os


@click.command()
def main() -> None:
    df = easy_run(*default_run_args, assign_params=False)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_pickle(f"data/simulation_output/{timestamp}.pkl.gz", compression="gzip")


if __name__ == "__main__":
    main()