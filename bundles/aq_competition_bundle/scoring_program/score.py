#!/usr/bin/env python

# Scoring program for the AutoML challenge
# Isabelle Guyon and Arthur Pesah, ChaLearn, August 2014-November 2016

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 

# Some libraries and options
import os
from sys import argv

import libscores
import yaml
from libscores import ls, filesep, mkdir, read_array, compute_all_scores, write_scores
import pandas as pd
import numpy as np

# Default I/O directories:
root_dir = "./"
default_input_dir = root_dir + "scoring_input_1_1"
default_output_dir = root_dir + "scoring_output"

# Debug flag 0: no debug, 1: show all scores, 2: also show version and listing of dir
debug_mode = 1

# Constant used for a missing score
missing_score = -0.999999

# Version number
scoring_version = 1.0


def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)


def _load_scoring_function():
    with open(_HERE('metric.txt'), 'r') as f:
        metric_name = f.readline().strip()
        return metric_name, getattr(libscores, metric_name)


# =============================== MAIN ========================================

if __name__ == "__main__":
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]
        # Create the output directory, if it does not already exist and open output files
    mkdir(output_dir)
    score_file = open(os.path.join(output_dir, 'scores.txt'), 'wb')
    html_file = open(os.path.join(output_dir, 'scores.html'), 'wb')

    # Get the metric
    metric_name, scoring_function = _load_scoring_function()

    # Get all the solution files from the solution directory
    solution_names = sorted(ls(os.path.join(input_dir, 'ref', '*.csv')))

    # Loop over files in solution directory and search for predictions with extension .predict having the same basename
    for i, solution_file in enumerate(solution_names):
        set_num = i + 1  # 1-indexed
        score_name = 'set%s_score' % set_num

        # Extract the dataset name from the file name
        basename = solution_file[-solution_file[::-1].index(filesep):-solution_file[::-1].index('.') - 1]

        try:
            # Get the last prediction from the res subdirectory (must end with '.csv')
            predict_file = ls(os.path.join(input_dir, 'res', '*.csv'))[-1]
            if (predict_file == []): raise IOError('Missing prediction file {}'.format(basename))
            predict_name = predict_file[-predict_file[::-1].index(filesep):-predict_file[::-1].index('.') - 1]

            # Get solution and prediction files here
            pd_solution = pd.read_csv(solution_file, sep=',')
            pd_prediction = pd.read_csv(predict_file, sep=',')
            
            # Check shape
            if (len(pd_solution) != len(pd_prediction)): raise ValueError(
                "Bad prediction shape {}".format(prediction.shape))

            # leftjoin sol and pred data
            pd_merged_all = pd.merge( pd_solution, pd_prediction, how='left', on=['submit_date','test_id'] )
            # pd_merged = ['submit_date', 'test_id', 'PM2.5_x', 'PM10_x', 'O3_x', 'PM2.5_y', 'PM10_y', 'O3_y']
            # filter missing value
            pd_merged_pm25 = pd_merged_all[['submit_date', 'test_id', 'PM2.5_x', 'PM2.5_y']].replace('', np.nan, regex=True).dropna(subset=['PM2.5_x']).replace(np.nan, 0, regex=True)
            pd_merged_pm10 = pd_merged_all[['submit_date', 'test_id', 'PM10_x', 'PM10_y']].replace('', np.nan, regex=True).dropna(subset=['PM10_x']).replace(np.nan, 0, regex=True)
            pd_merged_o3 = pd_merged_all[['submit_date', 'test_id', 'O3_x', 'O3_y']].replace('', np.nan, regex=True).dropna(subset=['O3_x']).replace(np.nan, 0, regex=True)

            try:
                # Compute the score prescribed by the metric file 
                score = scoring_function(solution, prediction)
                print(
                    "======= Set %d" % set_num + " (" + predict_name.capitalize() + "): score(" + score_name + ")=%0.12f =======" % score)
                html_file.write(
                    "======= Set %d" % set_num + " (" + predict_name.capitalize() + "): score(" + score_name + ")=%0.12f =======\n" % score)
            except:
                raise Exception('Error in calculation of the specific score of the task')

            if debug_mode > 0:
                scores = compute_all_scores(solution, prediction)
                write_scores(html_file, scores)

        except Exception as inst:
            score = missing_score
            print(
                "======= Set %d" % set_num + " (" + basename.capitalize() + "): score(" + score_name + ")=ERROR =======")
            html_file.write(
                "======= Set %d" % set_num + " (" + basename.capitalize() + "): score(" + score_name + ")=ERROR =======\n")
            print
            inst

        # Write score corresponding to selected task and metric to the output file
        score_file.write(score_name + ": %0.12f\n" % score)

    # End loop for solution_file in solution_names
        html_file.close()
    score_file.close()

    # Lots of debug stuff
    if debug_mode > 1:
        swrite('\n*** SCORING PROGRAM: PLATFORM SPECIFICATIONS ***\n\n')
        show_platform()
        show_io(input_dir, output_dir)
        show_version(scoring_version)

        # exit(0)
