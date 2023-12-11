# This file is only used for CLI implementation
# and it is not truly necessary to look into.

import argparse
import os
import shelve
import shutil

from .info import *


def main():
    if not os.path.exists("./dbdir"):
        os.makedirs("./dbdir")

    parser = argparse.ArgumentParser(
        prog="Reinforcement Learning Maze",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("-b", "--base",
                        choices=["graph", "board"],
                        help="Used for generating a base. Provide a base type when executing.")

    parser.add_argument("-s", "--size",
                        type=int,
                        nargs="+",
                        help="Used for specifying a base size. Provide an `int` for `graph` bases or `tuple(int, int)` for `board` bases.")

    parser.add_argument("-env", "--environment",
                        nargs=1,
                        type=float,
                        help="Used for generating an environment. Provide a `gamma` value when executing.")

    parser.add_argument("-a", "--agent",
                        action="store_true",
                        help="Used for generating an agent.")

    parser.add_argument("-cmpt", "--compute",
                        action="store_true",
                        help="Used for computing optimal V values for a given environment.")

    parser.add_argument("-i", "--info",
                        choices=["base", "bases",
                                 "env", "envs",
                                 "agent", "agents",
                                 "probs"],
                        help="Used for displaying information about the maze itself.")

    parser.add_argument("-d", "--delete",
                        action="store_true",
                        help="Delete everything you've created [[DANGEROUS]].")

    args = parser.parse_args()

    with shelve.open("./dbdir/db", writeback=True) as db:
        if args.base:
            if not args.size:
                raise Exception(
                    "You must specify size for the base you want!"
                )

            name = input(
                "Write a base name or press enter to set a default name: "
            )

            default_specs = [
                (7, lambda: RegularCell(-1)),
                (2, lambda: RegularCell(-10)),
                (2, lambda: WallCell()),
                (2, lambda: TerminalCell(-1)),
                (1, lambda: TeleportCell()),
            ]

            if "bases" not in db:
                db["bases"] = dict()

            match args.base:
                case "graph":
                    if len(args.size) != 1:
                        raise Exception(
                            "Provide not more or less than number of nodes in a graph!"
                        )

                    base = MazeGraph(
                        no_nodes=args.size[0], specs=default_specs)

                case "board":
                    if len(args.size) != 2:
                        raise Exception(
                            "Provide not more or less than number of rows and columns in a graph!"
                        )
                    base = MazeBoard(size=tuple(args.size),
                                     specs=default_specs)

            name = name if name != "" else "base" + str(len(db["bases"]))
            db["bases"][name] = base

        if args.environment:
            if "bases" not in db:
                raise Exception(
                    "You should have at least one base created before the environment itself!"
                )

            if "envs" not in db:
                db["envs"] = dict()

            ename = input(
                "Give this environment a name or press enter to set a default name: "
            )
            ename = ename if ename != "" else "env" + str(len(db["envs"]))

            bname = input(
                "Type a name of a base you want for this environment (it should be already created): "
            )
            base = db["bases"].get(bname)

            if not base:
                raise Exception(
                    "No bases with that name! Type '--info bases' for more."
                )

            env = MazeEnvironment(base, float(args.environment[0]))

            db["envs"][ename] = [env, bname]

        if args.agent:
            if "envs" not in db:
                raise Exception(
                    "Have at least one environment created before you create an agent!"
                )

            ename = input(
                "What environment would you like to use (should be already created): "
            )

            eb = db["envs"].get(ename)
            if not eb:
                raise Exception(
                    f"Environment named {ename} is not created yet!"
                )

            aname = input(
                "Type a name for agent or press enter to set a default name: ")

            if "agents" not in db:
                db["agents"] = dict()

            aname = aname if aname != "" else "agent" + str(len(db["agents"]))

            db["agents"][aname] = [
                Agent(eb[0], actions=eb[0].get_actions()), ename]

        if args.compute:
            if "agents" not in db:
                raise Exception(
                    "First of all, create at least one agent!"
                )

            aname = input(
                "For what agent would you like to compute V values (should be already created): "
            )

            ae = db["agents"].get(aname)

            if not ae:
                raise Exception(
                    f"No agent named {aname}!"
                )

            _, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
            axes = axes.flatten()

            Info.draw_values(ae[0], ax=axes[0])
            axes[0].set_title("Initial V values")

            k = ae[0].env.compute_values()

            Info.draw_values(ae[0], ax=axes[1])
            axes[1].set_title(
                f"V values computed using Q values after {k} iterations.")

            Info.draw_policy(ae[0], "greedy_v", ax=axes[2])
            axes[2].set_title("Optimal policy using V values")

            Info.draw_policy(ae[0], "greedy_q", ax=axes[3])
            axes[3].set_title("Optimal policy using Q values")

            plt.show()

        if args.info:
            match args.info:

                case "base":
                    name = input(
                        "Provide a base name that you want to look at: ")

                    base = db["bases"].get(name)
                    if base:
                        _, axes = plt.subplots(
                            nrows=1, ncols=1, figsize=(10, 5))
                        Info.draw_base(base, ax=axes)
                        plt.show()
                    else:
                        print(
                            f"No base named '{name}'!"
                        )

                case "bases":
                    if "bases" not in db:
                        raise Exception(
                            "You haven't created any bases yet!"
                        )

                    to_print = list()

                    for bname in db["bases"]:
                        base = db["bases"][bname]
                        btype = base.__class__.__name__
                        bsize = base.size

                        to_print.append([bname, btype, bsize])

                    print(tabulate(to_print, headers=[
                          "Base name", "Base type", "Base size"], tablefmt="rst"))

                case "env":
                    if "envs" not in db:
                        raise Exception(
                            "You haven't created any environments yet!"
                        )

                        ename = input(
                            "What base would you like to look into?: "
                        )

                        eb = db["envs"][ename]

                        if not eb:
                            raise Exception(
                                f"No base named {ename}!"
                            )

                        bname = eb[1]
                        btype = db["bases"][bname].__class__.__name__
                        egamma = eb[0].gamma

                        print(tabulate([ename, bname, btype, egamma],
                                       headers=["Environment name", "Base name",
                                                "Base type", "\u03B3 value"],
                                       tablefmt="rst"
                                       )
                              )

                case "envs":
                    if "envs" not in db:
                        raise Exception(
                            "You haven't created any environments yet!"
                        )

                    to_print = list()

                    for ename in db["envs"]:
                        bname = db["envs"][ename][1]
                        btype = db["bases"][bname].__class__.__name__
                        egamma = db["envs"][ename][0].gamma

                        to_print.append([ename, bname, btype, egamma])

                    print(tabulate(to_print,
                                   headers=["Environment name", "Base name",
                                            "Base type", "\u03B3 value"],
                                   tablefmt="rst"
                                   ))

                case "agent":
                    if "agents" not in db:
                        raise Exception(
                            "You haven't created any agents yet!"
                        )

                    aname = input(
                        "What agent would you like to look into?: "
                    )

                    ae = db["agents"][aname]

                    if not ae:
                        raise Exception(
                            f"No agent named {aname}!"
                        )

                    ename = ae[1]

                    print(tabulate([aname, ename],
                                   headers=["Agent name", "Environment name"],
                                   tablefmt="rst"
                                   ))

                case "agents":
                    if "agents" not in db:
                        raise Exception(
                            "You haven't created any agents yet!"
                        )

                    to_print = list()

                    for aname in db["agents"]:
                        ename = db["agents"][aname][1]
                        to_print.append([aname, ename])

                    print(tabulate(to_print,
                                   headers=["Agent name", "Environment name"],
                                   tablefmt="rst"
                                   ))

                case "probs":
                    if "envs" not in db:
                        raise Exception(
                            "No environments created! Create at least one environment."
                        )

                    ename = input(
                        "For what environment would you like to display properties? Provide a name: "
                    )

                    eb = db["envs"].get(ename)

                    if not eb:
                        raise Exception(
                            f"No environment named '{ename}'!"
                        )

                    Info.print_probabilities(eb[0])

        if args.delete:
            if os.path.exists("./dbdir"):
                shutil.rmtree("./dbdir")
