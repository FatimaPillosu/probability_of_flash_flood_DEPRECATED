import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, average_precision_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.verif_scores import (contingency_table_probabilistic, 
                                                      precision,
                                                      hit_rate, 
                                                      false_alarm_rate,
                                                      reliability_diagram,
                                                      aroc_trapezium
                                                      )
from matplotlib.ticker import FuncFormatter

########################################################################################
# CODE DESCRIPTION
# 23_prob_ff_hydro_short_fc_verif.py plots the verification results over the training and verification dataset.
# The following scores were computed:
#     - reliability diagram (breakdown reliability score)
#     - frequency bias (overall score)
#     - roc curve (breakdown discrimination ability)
#     - area under the roc curve (overall discrimination ability)
#     - precision-recall curve (breakdown score for imbalanced datasets)
#     - area under the precision-recall curve (overall performance)

# Usage: python3 23_prob_ff_hydro_short_fc_verif.py

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# ml_trained_list (list of strings): names of the models to train. Valid values are:
#                                                                 - random_forest_xgboost
#                                                                 - random_forest_lightgbm
#                                                                 - gradient_boosting_xgboost
#                                                                 - gradient_boosting_lightgbm
#                                                                 - gradient_boosting_catboost
#                                                                 - feed_forward_keras
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the verification results of the model trainings.
# dir_out (string): relative path of the directory containing the plots for the considered verification scores.

########################################################################################
# INPUT PARAMETERS
ml_trained_list = ["gradient_boosting_xgboost"]
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/processed/13_prob_ff_hydro_short_fc_retrain_best_kfold"
dir_out = "data/plot/29_paper_verif_plots_short_fc"
##############################################################################################################


# Creating the verification plots
for loss_func in ["bce"]:

      for eval_metric in ["auc"]:

            print(f"\nCreating verification plots for loss_fun = {loss_func}, and eval_metric = {eval_metric}")

            # Creating the input/output directory
            dir_in_temp = f'{git_repo}/{dir_in}/{loss_func}/{eval_metric}'
            dir_out_temp = f'{git_repo}/{dir_out}/{loss_func}/{eval_metric}'
            os.makedirs(dir_out_temp, exist_ok=True)

            # Initialising the variables storing the overall scores
            auprc_test_all = []
            aroc_test_all = []
            fb_test_all = []

            # Initialising the variables containing the distribution of forecast probabilities
            fc_prob_all = []
            fc_all = []
            fc_prob_max_all = []

            # Computing the verification scores
            for ml_trained in ml_trained_list:

                  print(f" - Plots for {ml_trained}")

                  # Reading the predictions and observations
                  fc_prob_test = np.load(f"{dir_in_temp}/{ml_trained}/fc_test.npy") * 100
                  obs_test = np.load(f"{dir_in_temp}/{ml_trained}/obs_test.npy")
                  prob_thr = np.load(f"{dir_in_temp}/{ml_trained}/best_thr.npy") * 100
                  fc_test = fc_prob_test >= prob_thr
                  fc_all.append(np.sum(fc_test) / len(fc_test) * 100)
                  fc_prob_all.append(fc_prob_test)
                  fc_prob_max_all.append(np.max(fc_prob_test))

                  # Computing the contingency table
                  h_test, fa_test, m_test, cn_test = contingency_table_probabilistic(obs_test, fc_prob_test, 100)
                  
                  # Plotting the precision-recall curve
                  plt.figure(figsize=(6.5, 6))
                  p_test = precision(h_test, fa_test)
                  hr_test = hit_rate(h_test, m_test)
                  ref_test = np.sum(obs_test) / len(obs_test)
                  auprc_test_all.append(average_precision_score(obs_test, fc_prob_test))
                  plt.plot(hr_test, p_test, "-o", color = "#00B0F0", lw = 3, ms=5)
                  plt.plot([0,1], [ref_test, ref_test], "-", color = "#333333", lw = 2)
                  plt.xlabel("Recall", color = "#333333", fontsize = 28, labelpad = 15)
                  plt.ylabel("Precision", color = "#333333", fontsize = 28, labelpad = 15)
                  plt.tick_params(axis='x', colors='#333333', labelsize=28)
                  plt.tick_params(axis='y', colors='#333333', labelsize=28)
                  plt.xticks(np.arange(0, 1.01, 0.2))
                  plt.yticks(np.arange(0, 0.41, 0.1))
                  fmt = FuncFormatter(lambda val, pos: f"{val:.1f}".lstrip("0") if abs(val) < 1 else f"{val:.1f}")
                  ax = plt.gca()
                  ax.xaxis.set_major_formatter(fmt)
                  ax.yaxis.set_major_formatter(fmt)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-0.02,1.02])
                  plt.ylim([-0.02,0.42])
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/pr_curve_{ml_trained}.png', dpi=1000)
                  plt.close()

                  # Plotting the ROC curve - Trapezium and Continuous
                  plt.figure(figsize=(6.5, 6))           
                  hr_test = hit_rate(h_test, m_test)
                  far_test = false_alarm_rate(fa_test, cn_test)
                  aroc_test = aroc_trapezium(hr_test, far_test)
                  plt.plot(far_test, hr_test, "-o", color = "#00B0F0", lw = 3, ms=5, label = f"{aroc_test:.3f}")
                  far_test_c, hr_test_c, thr_roc = roc_curve(obs_test, fc_prob_test)
                  aroc_test_c = auc(far_test_c, hr_test_c)
                  aroc_test_all.append(aroc_test_c)
                  plt.plot(far_test_c, hr_test_c, "-", color = "#00B0F0", lw = 1, ms=2, label = f"{aroc_test_c:.3f}")
                  plt.plot([0,1], [0, 1], "-", color = "#333333", lw = 1)
                  plt.xlabel("False Alarm Rate", color = "#333333", fontsize = 28, labelpad = 15)
                  plt.ylabel("Hit Rate", color = "#333333", fontsize = 28)
                  plt.xticks(np.arange(0, 1.01, 0.2))
                  plt.yticks(np.arange(0, 1.01, 0.2))
                  plt.tick_params(axis='x', colors='#333333', labelsize=28)
                  plt.tick_params(axis='y', colors='#333333', labelsize=28)
                  fmt = FuncFormatter(lambda val, pos: f"{val:.1f}".lstrip("0") if abs(val) < 1 else f"{val:.1f}")
                  ax = plt.gca()
                  ax.xaxis.set_major_formatter(fmt)
                  ax.yaxis.set_major_formatter(fmt)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-0.02,1.02])
                  plt.ylim([-0.02,1.02])
                  plt.legend(title = "AROC", title_fontsize=24, fontsize=24, frameon=False, loc='lower right')
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/roc_curve_{ml_trained}.png', dpi=1000)
                  plt.close()

                  # Plotting the reliability diagram
                  fig, ax = plt.subplots(figsize=(6.5, 6))
                  mean_prob_fc_test, mean_freq_obs_test, sharpness_test = reliability_diagram(obs_test, fc_prob_test)
                  plt.plot(mean_prob_fc_test, mean_freq_obs_test * 100, "-o", color = "#00B0F0", lw = 3, ms=5)
                  plt.plot([0,100], [0, 100], color = "#333333", lw = 1)
                  plt.xlabel("Forecast probability", color = "#333333", fontsize = 28, labelpad = 15)
                  plt.ylabel("Observed frequency", color = "#333333", fontsize = 28, labelpad = 20)
                  plt.tick_params(axis='x', colors='#333333', labelsize=28)
                  plt.tick_params(axis='y', colors='#333333', labelsize=28)
                  ticks = np.arange(0, 101, 5)
                  labels = [str(t) if t % 10 == 0 else '' for t in ticks]
                  plt.xticks(ticks, labels)
                  plt.grid(axis='y', linewidth=0.5, color='gainsboro')
                  plt.xlim([-1,31])
                  plt.ylim([-1,31])
                  plt.tight_layout()
                  plt.savefig(f'{dir_out_temp}/reliability_diagram_{ml_trained}.png', dpi=1000)
                  plt.close()

                  # Computing the frequency bias
                  fb_test_all.append( np.sum(fc_test) / np.sum(obs_test))

            print(auprc_test_all)
            print(aroc_test_all)
            print(fb_test_all)