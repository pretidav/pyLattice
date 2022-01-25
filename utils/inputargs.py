import argparse
import logging

class Parser():
    def __init__(self):
        self.args = self.parser()
        self.grid = self.parse_latt(par=self.args['grid'])
        self.mpigrid = self.parse_latt(par=self.args['mpigrid'])
        self.seed    = int(self.args['seed'])
        self.debug   = self.args['debug']
        self.logfile = self.args['logfile']
        self.beta = float(self.args['beta'])
        self.steps = int(self.args['steps'])

    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--grid', help='XxYxZx...', default='4x4')
        parser.add_argument('--mpigrid', help="XxY...", default="1x2")
        parser.add_argument('--steps', help='XxYxZx...', default=100)
        parser.add_argument('--beta', help="XxY...", default=0.5)
        parser.add_argument('--seed', help="XxY...", default=0)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--logfile', action='store_true')

        return vars(parser.parse_args())

    def parse_latt(self, par):
        return [int(a) for a in str(par).split('x')]

class Logger():
    def __init__(self,debug,logfile):
        self.debug = debug
        self.logfile = logfile
        self.log = self.get_logger(debug=self.debug, logfile=self.logfile)
    
    def get_logger(self, debug, logfile):
        if logfile:
            logging.basicConfig(level=logging.WARNING,
                                filename='split.log',
                                format='%(asctime)s %(levelname)s:  %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING,
                                format='%(asctime)s %(levelname)s:  %(message)s')
        logging.getLogger().setLevel(level=logging.INFO)
        if debug:
            logging.getLogger().setLevel(level=logging.DEBUG)
            logging.getLogger('matplotlib.font_manager').disabled = True
        return logging.getLogger(__name__)