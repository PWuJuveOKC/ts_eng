import pandas as pd
import numpy as np

dat = pd.read_csv('Data/nonstat_ar_TS_180228_2200.csv')

renames = ['timelength',  'value_count_value_0.0', 'value_count_value_1.0', 'value_count_median','number_crossing_mean',
           'number_crossing_meansigma_lb', 'maximum', 'mean', 'median', 'minimum','standard_deviation', 'ratio_beyond_r_sigma_r_1',
           'ratio_beyond_r_sigma_r_7', 'kurtosis', 'skewness', 'relative_quantile_q_0.2', 'relative_quantile_q_0.8',
           'index_mass_quantile_q_0.2', 'index_mass_quantile_q_0.4','index_mass_quantile_q_0.6', 'index_mass_quantile_q_0.8',
           'symmetry_v', 'augmented_dickey_fuller_attr_pvalue', 'augmented_dickey_fuller_attr_teststat', 'binned_entropy_max_bins_10',
           'energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_0','energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_1','energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_2',
           'energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_3','energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_4','energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_5',
           'energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_6','energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_7','energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_8',
           'energy_ratio_by_chunks_num_sgmts_10_sgmt_focus_9', 'number_peaks_r_n_1', 'number_peaks_r_n_5', 'number_peaks_r_n_50',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C0_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C1_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C2_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C3_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C4_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C5_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C6_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C7_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C8_bins10_f_agg_mean',
           'cwt_coeff_agg_widths_(2-5-10-20)_w_10_C9_bins10_f_agg_mean',
           'frequency_info_N_components', 'frequency_info_maxFreq',
           'frequency_info_maxPeriod', 'frequency_info_minFreq','number_cwt_peaks_n_5',
           'agg_autocorrelation_vlag_lag_0.25_f_agg_mean','agg_autocorrelation_vlag_lag_0.25_f_agg_var',
            'ar_coefficient_k_10_coeff_0','ar_coefficient_k_10_coeff_1', 'ar_coefficient_k_10_coeff_2',
            'ar_coefficient_k_10_coeff_3', 'ar_coefficient_k_10_coeff_4','partial_autocorrelation_lag_2', 'partial_autocorrelation_lag_3',
            'partial_autocorrelation_lag_4', 'partial_autocorrelation_lag_5',
            'partial_autocorrelation_lag_6', 'partial_autocorrelation_lag_7',
            'partial_autocorrelation_lag_8', 'partial_autocorrelation_lag_9',
            'change_quant_f_agg_mean_isabs_True_qh_0.2_ql_0.0','change_quant_f_agg_mean_isabs_True_qh_0.4_ql_0.2',
            'change_quant_f_agg_mean_isabs_True_qh_0.6_ql_0.4','change_quant_f_agg_mean_isabs_True_qh_0.8_ql_0.6',
            'change_quant_f_agg_mean_isabs_True_qh_1.0_ql_0.8','change_quant_f_agg_var_isabs_True_qh_0.2_ql_0.0',
            'change_quant_f_agg_var_isabs_True_qh_0.4_ql_0.2','change_quant_f_agg_var_isabs_True_qh_0.6_ql_0.4',
            'change_quant_f_agg_var_isabs_True_qh_0.8_ql_0.6','change_quant_f_agg_var_isabs_True_qh_1.0_ql_0.8',
            'time_reversal_asymmetry_statistic_lag_2','c3_lag_1', 'c3_lag_3',
            'friedrich_coefficients_t_norm_1_m_3_r_30_coeff_0','friedrich_coefficients_t_norm_1_m_3_r_30_coeff_1',
            'friedrich_coefficients_t_norm_1_m_3_r_30_coeff_2','friedrich_coefficients_t_norm_1_m_3_r_30_coeff_3',
            'max_langevin_fixed_point_t_m_3_r_30','linTrend_attr_icpt', 'linTrend_attr_pvalue',
            'linTrend_attr_rvalue', 'linTrend_attr_slope','linTrend_attr_stderr'
           ]

dat_new = dat[renames]

dat_new.to_csv('Data/nonstat_285_heat.csv',index=None)
