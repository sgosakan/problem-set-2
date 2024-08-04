'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import part1_etl as etl
import part2_preprocessing as preprocessing
import part3_logistic_regression as logistic_regression
import part4_decision_tree as decision_tree
import part5_calibration_plot as calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instantiate ETL, saving the two datasets in `./data/`
    etl.etl()

    # PART 2: Call functions/instantiate objects from preprocessing
    preprocessing.preprocess()

    # PART 3: Call functions/instantiate objects from logistic_regression
    logistic_regression.logistic_regression()

    # PART 4: Call functions/instantiate objects from decision_tree
    decision_tree.decision_tree()

    # PART 5: Call functions/instantiate objects from calibration_plot
    calibration_plot.calibration()


if __name__ == "__main__":
    main()