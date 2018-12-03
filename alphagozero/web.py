import logging
from typing import Dict, Any
import numpy as np
from flask import Flask, json
from datetime import timedelta
from functools import update_wrapper

from flask import current_app, request, make_response

from alphagozero.MCTS import MCTS
from alphagozero.game import Game
from alphagozero.nnet import NNet

def crossdomain(origin=None, methods=None, headers=None, max_age=21600, attach_to_all=True, automatic_options=True) -> Callable[[Any], Any]: #type: ignore
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

class WebServer(object):

    def __init__(self, game: Game, nnet: NNet, checkpoint_folder: str, c_puct: int, num_mcst_sims: int):
        self.nnet = nnet
        logging.info("Restoring network.")
        #self.nnet.load_checkpoint(folder=checkpoint_folder, filename="best.pth.tar")
        self.nnet.load_checkpoint(folder=checkpoint_folder, filename="temp.pth.tar")
        self.game = game
        self.mcts = MCTS(game=game, nnet=self.nnet, cpuct=c_puct, num_mcst_sims=num_mcst_sims, root_noise=False, board_size = 19)

    def start_web_server(self) -> None:
        # Start Flask Server.. (if web)
        app = Flask(__name__)

        @app.route('/get_move', methods=["POST"])
        #@crossdomain(origin='*')
        def get_moves() -> Dict[str, int]:
            board = request.form["board"]

            board = list(map(lambda y: int(y), filter(lambda x: x != ",", list(board))))
            board = np.asarray(board).reshape((19, 19))

            print(list(board))

            canonicalBoard = self.game.getCanonicalForm(board, player=1)

            pi = self.mcts.getActionProb(canonicalBoard,
                                        temp=1,
                                        current_self_play_iteration=0)

            print("Pi", ",".join(map(str, pi)))

            action = np.random.choice(len(pi), p=pi)

            print("Action", action)

            x = action % 19
            y = action // 19
            return json.dumps({"x": x, "y": y})

        app.run()
