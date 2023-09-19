## Python standard libraries
from __future__ import print_function
import os
import numpy as np
import pandas as pd


def add_to_excel(table, writer, model_names, metrics_names, run=1, overall:str=None):
    """
    Add a table to the excel spreadsheet
    Args:

    Returns: None
    """
    if overall == "avg_dev":
        table_avg = np.nanmean(table,axis=-1)
        table_std = np.nanstd(table,axis=-1)
        DF_avg = pd.DataFrame(table_avg, index=model_names, columns=metrics_names)
        DF_std = pd.DataFrame(table_std, index=model_names, columns=metrics_names)
        DF_avg.to_excel(writer, sheet_name='Overall Avg')
        DF_std.to_excel(writer, sheet_name='Overall Std')
    # elif overall == "best":
    #     DF = pd.DataFrame(table,index=model_names,columns=metrics_names)
    #     DF.colummns = metrics_names
    #     DF.index = model_names
    #     DF.to_excel(writer,sheet_name='Overall Best'.format(fold+1))
    else:
        DF = pd.DataFrame(table, index=model_names, columns=metrics_names)
        DF.columns = metrics_names
        DF.index = model_names
        DF.to_excel(writer, sheet_name='Run_{}'.format(run+1))


#Compute desired metrics and save to excel spreadsheet
def fill_metrics_spreadsheet(metrics, model_names, save_dir,
                             file_name="hyperpri_eval"):
    """
    Fill a spreadsheet with saved metrics for each run.
    This outputs pages for Runs, Average, Best, and Deviation.
    Args:
        metrics: list of dictionaries corresponding to each training run
        model_names: list of strings that represent the evaluated models
        save_dir: string specifying where to save the spreadsheet
        file_name: str of the saved filename (w/o extension). Default - 'hyperpri_eval'
    """
    metric_names = list(metrics[0][list(model_names)[0]].keys())

    #Intialize validation and test arrays
    eval_table = np.zeros((len(model_names),
                           len(metric_names),
                           len(metrics)))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    metrics_writer = pd.ExcelWriter(save_dir + f'/{file_name}.xlsx', engine='xlsxwriter')

    # Initialize the histogram model for this run
    for run in range(0, len(metrics)):
        for model_num, model in enumerate(model_names):
            for metric_num, metric_key in enumerate(metric_names):
                eval_table[model_num, metric_num, run] = metrics[run][model][metric_key]
            print('Run {}'.format(run + 1))

        #Add metrics to spreadsheet
        add_to_excel(eval_table[:, :, run], metrics_writer, model_names, metric_names, run=run)

    #Add metrics to spreadsheet
    add_to_excel(eval_table[:, :, run], metrics_writer, model_names, metric_names, run=run)

    #Compute average and std across folds
    add_to_excel(eval_table, metrics_writer, model_names, metric_names, overall="avg_dev")
    
    #Save spreadsheets
    metrics_writer.save()
