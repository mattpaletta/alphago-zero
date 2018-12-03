#original code https://github.com/suragnair/alpha-zero-general

import argparse
import yaml
from typing import Dict, Union

class Config(object):
    def get_args(self) -> argparse.Namespace:

        DATA_PATH = "argparse.yml"

        with open(DATA_PATH, "r") as file:
            configs = yaml.load(file)

        arg_lists = []
        parser = argparse.ArgumentParser()


        # Dynamically populate runtime arguemnts.
        for g_name, group in configs.items():
            arg = parser.add_argument_group(g_name)
            arg_lists.append(arg)

            for conf in group.keys():
                arg.add_argument("--" + str(conf), **group[conf])


        parsed, unparsed = parser.parse_known_args()

        return parsed
