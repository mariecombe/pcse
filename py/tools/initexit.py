#!/usr/bin/env python
# initexit.py

"""
.. module:: initexit
.. moduleauthor:: Wouter Peters 

Revision History:
File created on 13 May 2009.

"""
import logging
import os
import sys
import glob
import shutil
import copy
import getopt
import cPickle
from string import join

def start_logger(level=logging.INFO):
    """ start the logging of messages to screen"""

# start the logging basic configuration by setting up a log file

    logging.basicConfig(level=level,
                        format=' [%(levelname)-7s] (%(asctime)s) py-%(module)-20s : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def parse_options():
    """ 
    Function parses options from the command line and returns the arguments as a dictionary.
    Accepted command line arguments are:

    ========  =======
    Argument  Meaning
    ========  =======
    -v        verbose output in log files
    -h        display help
    -r        start a simulation by recovering from a previous crash
    -t        start a simulation by transitioning from 25 to 34 layers in December 2005 (od meteo)
    ========  =======

    """

# Parse keywords, the only option accepted so far is the "-h" flag for help

    opts = []
    args = []
    try:                                
        opts, args = getopt.gnu_getopt(sys.argv[1:], "-v")
    except getopt.GetoptError, msg:           
        logging.error('%s' % msg)
        sys.exit(2)      

    for options in opts:
        options = options[0].lower()
        if options == '-v':
            logging.info('-v flag specified on command line: extra verbose output')
            logging.root.setLevel(logging.DEBUG)

    if opts: 
        optslist = [item[0] for item in opts]
    else:
        optslist = []

# Parse arguments and return as dictionary

    arguments = {}
    for item in args:
        #item=item.lower()

# Catch arguments that are passed not in "key=value" format

        if '=' in item:
            key, arg = item.split('=')
        else:
            logging.error('%s' % 'Argument passed without description (%s)' % item)
            raise getopt.GetoptError, arg

        arguments[key] = arg


    return optslist, arguments

if __name__ == "__main__":
    pass

