import argparse


class Parser():
    def __init__(self):
        self.args = self.parser()
        self.grid = self.parse_latt(par=self.args['grid'])
        self.mpigrid = self.parse_latt(par=self.args['mpigrid'])

    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--grid', help='XxYxZx...', default='4x4')
        parser.add_argument('--mpigrid', help="XxY...", default="1x2")
        return vars(parser.parse_args())

    def parse_latt(self, par):
        return [int(a) for a in str(par).split('x')]
