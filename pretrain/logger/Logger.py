import logging 
import sys
import os

class Logger(object):
    def __init__(self, output_dir_path):
        self.output_dir_path = output_dir_path

        # create custom logger
        self.log = logging.getLogger('log')
        self.log.setLevel(logging.INFO)

        # create handlers
        stdout_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(os.path.join(self.output_dir_path, 'log.txt'))
        
        # create formatting and add it to handlers
        stdout_format = logging.Formatter('%(asctime)s %(message)s')
        file_format = logging.Formatter('%(asctime)s %(message)s')
        stdout_handler.setFormatter(stdout_format)
        file_handler.setFormatter(file_format)

        # add handlers to the logger
        self.log.addHandler(stdout_handler)
        self.log.addHandler(file_handler)

    def info(self, msg):
        self.log.info(msg)