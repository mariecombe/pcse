#!/usr/bin/env python
# maunaloa.py

"""
Author : peters 

Revision History:
File created on 06 Sep 2010.

"""

import logging

from py.baseclasses.platform import Platform, std_joboptions

class CartesiusPlatform(Platform):
    def __init__(self):

        self.ID = 'cartesius'    # the identifier gives the platform name
        self.version = '1.0'     # the platform version used

    def give_blocking_flag(self):
        return ""

    def give_queue_type(self):
        return "foreground"

    def get_job_header(self, joboptions={}, block=False):
        """ 
        Returns the job template for a given computing system, and fill it with options from the dictionary provided as argument.
        The job template should return the preamble of a job that can be submitted to a queue on your platform, 
        examples of popular queuing systems are:
            - SGE
            - MOAB
            - XGrid
            -

        A list of job options can be passed through a dictionary, which are then filled in on the proper line,
        an example is for instance passing the dictionary {'account':'co2'} which will be placed 
        after the ``-A`` flag in a ``qsub`` environment.

        An extra option ``block`` has been added that allows the job template to be configured to block the current
        job until the submitted job in this template has been completed fully.
        """

        template = """#!/bin/bash \n""" + \
                   """## \n""" + \
                   """## This is a set of dummy names, to be replaced by values from the dictionary \n""" + \
                   """## Please make your own platform specific template with your own keys and place it in a subfolder of the da package.\n """ + \
                   """## \n""" + \
                   """#SBATCH -J jobname \n""" + \
                   """#SBATCH -p jobqueue \n""" + \
                   """#SBATCH -n jobnodes \n""" + \
                   """#SBATCH -t jobtime \n""" + \
                   """#SBATCH -o joblog \n""" + \
                   """module load python/2.7.11\n""" + \
                   """module load nco\n""" + \
		   """\n"""

        if 'depends' in joboptions:
            template += """#$ -hold_jid depends \n"""

        # First replace from passed dictionary
        for k, v in joboptions.iteritems():
            while k in template:
                template = template.replace(k, v)

        # Fill remaining values with std_options
        for k, v in std_joboptions.iteritems():
            while k in template:
                template = template.replace(k, v)

        return template

    def submit_job(self, jobfile, joblog=None, block=False):
        """ This method submits a jobfile to the queue, and returns the queue ID """
        import subprocess


        #cmd     = ["llsubmit","-s",jobfile]
        #msg = "A new task will be started (%s)"%cmd  ; logging.info(msg)

        if block:
            cmd = ["salloc",'-n',std_joboptions['jobnodes'],'-t',std_joboptions['jobtime'], jobfile]
            logging.info("A new task will be started (%s)" % cmd)
            output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]
            logging.info(output)
            print 'output', output
            jobid = output.split()[-1]             
            print 'jobid', jobid
        else:
            cmd = ["sbatch", jobfile]
            logging.info("A new job will be submitted (%s)" % cmd)
            output = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0]  ; logging.info(output)
            jobid = output.split()[-1]
            
        return jobid



if __name__ == "__main__":
    pass
