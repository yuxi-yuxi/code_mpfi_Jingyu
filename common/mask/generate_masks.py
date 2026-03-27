# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 14:32:59 2025

@author: Jingyu Cao
@contributor: Yingxue Wang
"""
import os
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import skew, pearsonr
from scipy.ndimage import gaussian_filter1d

from common.utils_basic import run_of_ones
from common.mask.neuropil_mask import fiber_block_to_neuropil_masks
# from common.mask.utils_mask import load_axon_mask, generate_adaptive_membrane_mask, plot_membrane_mask, axon_mask_dilation, axon_mask_dilation_grid_free, dlight_regressor_mask_grid_free, dlight_regressor_mask
import common.mask.utils_mask as utl


classification_thresholds = {
        'min_soma_roi_npix': 100, # number of pixels in each ROI
        'max_soma_roi_npix': 600,
        'compactness_threshold_min': 0.1, # compactness calculated by 4 * pi * npix / (perimeter**2) default: 0.2
        'hollowness_threshold_min': 0.25, # whether the ROI & global_geco_mask is hollow default: 0.25
        'area_threshold': 2000, # area of the square box around the ROI
        'aspect_ratio_max': 3, 
        'skewness_threshold': 0.8, # skewness of F-0.7*Fneu,
        'min_act_frames': 15, # for GCaMP8s, half decay time ~ 307 ms (~10 frames)
        'max_act_frames': 150, # to exclude overexpression rois
    }


    
def generate_axon_dlight_masks_wrap(
                        mean_img_red, mean_img_green,
                        path_result=None,
                        dilation_steps = (0, 2, 4, 6, 8, 10),
                        grid_size = 16,
                        dilation_method = 'binary',
                        constrain_to_grid = True,
                        visualize_neu = False,
                        visualize_dilation = False,
                        visualize_regressor = False
                        ):
    path_result = Path(path_result)
    axon_mask = utl.load_axon_mask(path_result)
    # create enchanced dlight mask and visualize
    print('creating enchanced dlight mask and visualize...')
    base_mask, global_dlight_mask_enhanced, fig = utl.generate_adaptive_membrane_mask(
        mean_img_green,
        gaussian_sigma=1,
        adaptive_block_size=21,
        valley_radius=10,        # ~3–4 µm footprint
        uniformity_thresh=3.5,      # ↑ for stricter "neropil-like"
        z_tau=3.8,              # ↑ for more selective neuropil
        min_region_size=150,
        hole_size_threshold=150,
    )
    np.save(path_result / 'global_dlight_mask_enhanced.npy', global_dlight_mask_enhanced)
    plt.savefig(path_result / 'global_dlight_mask_enhanced_generation.png', dpi=200)
    plt.close()
    utl.plot_membrane_mask(mean_img_green, 
                       base_mask, global_dlight_mask_enhanced,
                       path_result/'dlight_mask_enhanced.png')
    
    # choose which filber and dlight mask to use
    global_dlight_mask = global_dlight_mask_enhanced
    global_axon_mask = axon_mask
    print('!!! Using predefined axon mask')
    
    # create dilated axon masks and neuropil masks
    print('creating dilated axon masks and neuropil masks...')
    if constrain_to_grid == True:
        utl.axon_mask_dilation(global_axon_mask, global_dlight_mask, mean_img_red,
                               dilation_steps,
                               method=dilation_method,
                               output_dir=path_result,
                               constrain_to_grid=constrain_to_grid,
                               grid_size=16,)
    
        utl.dlight_regressor_mask (path_result, 
                            mean_img_red,
                            dilation_steps,
                            neu_pix=3,
                            output_dir=path_result,
                            constrain_to_grid=constrain_to_grid,
                            grid_size=16)
    else:
        utl.axon_mask_dilation_grid_free(global_axon_mask, global_dlight_mask, mean_img_red,
                               dilation_steps,
                               method=dilation_method,
                               output_dir=path_result,
                               grid_size=16,
                               visualize=visualize_dilation,
                               fig_out=path_result)
        utl.dlight_regressor_mask_grid_free (path_result, 
                            mean_img_red,
                            dilation_steps,
                            neu_pix=3,
                            output_dir=path_result,
                            grid_size=16,
                            visualize=visualize_regressor,
                            fig_out=path_result)
    
    
    for k in dilation_steps:
        global_axon_mask  = np.load(path_result/'dilated_global_axon_k=0.npy')
        # h, w = global_axon_mask.shape
        # n_y = (h + grid_size - 1) // grid_size
        # n_x = (w + grid_size - 1) // grid_size
        # if k==0:
        #     current_axon_mask = utl.split_map_to_grids_full(global_axon_mask, n_y, n_x)
        # else:
        current_axon_mask = np.load(path_result/f'dilated_global_axon_k={k}.npy')
        fiber_block_to_neuropil_masks(current_axon_mask, global_axon_mask, 
                                      grid_size=grid_size,
                                      constrain_to_grid=constrain_to_grid,
                                      output_dir=path_result/f'fiber_neuropil_masks_dilation_k={k}.npy',
                                      visualize=visualize_neu,
                                      fig_out=path_result
                                      )

    
def generate_geco_dlight_masks_wrap(mean_img_red, mean_img_green,
                             path_result,
                             ):
    # create enchanced dlight mask and visualize
    print('creating enchanced dlight mask and visualize...')
    base_mask, global_dlight_mask_enhanced, fig = utl.generate_adaptive_membrane_mask(
        mean_img_green,
        gaussian_sigma=1,
        adaptive_block_size=21,
        valley_radius=10,        # ~3–4 µm footprint
        uniformity_thresh=3.5,      # ↑ for stricter "neropil-like"
        z_tau=3.8,              # ↑ for more selective neuropil
        min_region_size=150,
        hole_size_threshold=150,
    )
    np.save(path_result / 'global_dlight_mask_enhanced.npy', global_dlight_mask_enhanced)
    plt.savefig(path_result / 'global_dlight_mask_enhanced_generation.png', dpi=200)
    plt.close()
    utl.plot_membrane_mask(mean_img_green, 
                       base_mask, global_dlight_mask_enhanced,
                       path_result / 'dlight_mask_enhanced.png')


    # create enchanced geco mask and visualize
    print('creating enchanced geco mask and visualize...')
    base_mask, global_geco_mask_enhanced, fig = utl.generate_adaptive_membrane_mask(
        mean_img_red,
        gaussian_sigma=1,
        adaptive_block_size=21,
        valley_radius=10,        # ~3–4 µm footprint
        uniformity_thresh=3.5,      # ↑ for stricter "neropil-like"
        z_tau=3.8,              # ↑ for more selective neuropil
        min_region_size=150,
        hole_size_threshold=150,
    )
    np.save(path_result / 'global_geco_mask_enhanced.npy', global_geco_mask_enhanced)
    plt.savefig(path_result / 'global_geco_mask_enhanced_generation.png', dpi=200)
    plt.close()
    utl.plot_membrane_mask(mean_img_red, 
                       base_mask, global_geco_mask_enhanced,
                       path_result / 'geco_mask_enhanced.png')
    

def extract_axon_dlight_masks(rec, p_mask_result, dilation_steps,
                              constrain_to_grid=True,
                              visualize_neu = False,
                              visualize_dilation = False,
                              visualize_regressor = False):
    p_rec = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}".format(rec[:5], 
                                                        rec[:-3], 
                                                        rec[-2:],
                                                        'RegOnly')
    p_data = p_rec+r'\suite2p\plane0'
    suite2p_ops = np.load(p_data+r'\ops.npy', allow_pickle=True).item()
    mean_img_red = suite2p_ops['meanImg_chan2']
    mean_img_green = suite2p_ops['meanImg']
    
    generate_axon_dlight_masks_wrap(mean_img_red, mean_img_green,
                                    p_mask_result,
                                    dilation_steps = dilation_steps,
                                    dilation_method = 'binary',
                                    constrain_to_grid = constrain_to_grid,
                                    visualize_neu = visualize_neu,
                                    visualize_dilation = visualize_dilation,
                                    visualize_regressor = visualize_regressor)
 
def select_geco_rois(mean_img, F_corr, stat_array, path_result='',
                      thresholds=classification_thresholds):
    # mean_img: reference mean image of GCaMP
    # F_corr: np array, (n_rois, n_frames) F_raw-0.7*Fneu corrected F trace fo all rois
    print('selecting active soma masks...')
    global_soma_mask = np.load(path_result / 'global_geco_mask_enhanced.npy') 
    # anatomical features
    roi_npix    = np.array([r['npix'] for r in stat_array])
    hollowness_scores = utl.calculate_rois_hollowness(stat_array=stat_array, soma_mask=global_soma_mask)
    compactness, area, aspect_ratio_all, _ = utl.calculate_rois_compactness(stat_array=stat_array)
    # activity
    skewness   = skew(F_corr, axis=-1)
    all_F_mean = np.nanmean(F_corr, axis=-1) # (n_rois, )
    all_F_std  = np.nanstd(F_corr, axis=-1)  # (n_rois, )
    act_thresh = all_F_mean+2*all_F_std
    conti_act_frames = run_of_ones(F_corr > act_thresh[:, None])
    conti_act_frames_max = np.array([np.max(x) if len(x)>0 else np.nan for x in conti_act_frames])
    # correlation = pearsonr(F_raw, F_neu, axis=-1)[0]   

    is_soma = (
        (roi_npix > thresholds['min_soma_roi_npix']) &
        (roi_npix < thresholds['max_soma_roi_npix']) &
        (area < thresholds['area_threshold']) &
        (compactness > thresholds['compactness_threshold_min']) & # Corrected logic for compactness
        (aspect_ratio_all < thresholds['aspect_ratio_max'])&
        (hollowness_scores > thresholds['hollowness_threshold_min'])
    )
    is_active = ((skewness > thresholds['skewness_threshold'])&
                 (conti_act_frames_max > thresholds['min_act_frames'])&
                 (conti_act_frames_max < thresholds['max_act_frames']))
    is_active_soma = (is_soma)&(is_active)
    
    
    return is_soma, is_active, is_active_soma

def plot_geco_selection(F_corr, mean_img, suite2p_stat,
                        is_soma, is_active, is_active_soma,
                        out_dir_fig):
    n_rois = len(suite2p_stat)
    roi_map = np.zeros((n_rois, 512, 512))
    for idx, roi in enumerate(suite2p_stat):
        roi_map[idx, roi['ypix'], roi['xpix']] = random.randrange(1, 20)
    roi_map[roi_map<1] = np.nan
    fig, axs = plt.subplots(1, 3, figsize=(6, 2))
    mean_img = np.clip(mean_img, np.nanpercentile(mean_img, 2), 
                       np.nanpercentile(mean_img, 99))
    for ax in axs:
        ax.imshow(mean_img, cmap='gray')
        ax.axis('off')
    axs[0].imshow(np.nanmax(roi_map[is_soma], axis=0), cmap='tab20b', alpha=0.4)
    axs[0].set_title('soma')
    axs[1].imshow(np.nanmax(roi_map[~is_soma], axis=0), cmap='tab20b', alpha=0.4)
    axs[1].set_title('non-soma')
    axs[2].imshow(np.nanmax(roi_map[is_active_soma], axis=0), cmap='tab20b', alpha=0.4)
    axs[2].set_title('active soma')
    fig.tight_layout()
    # plt.show()
    plt.savefig(out_dir_fig, dpi=200)
    plt.close()
    
    
def extract_geco_dlight_masks(rec, p_masks, classification_thresholds=classification_thresholds):
    p_masks = Path(p_masks)
    p_rec = r"Z:\Jingyu\2P_Recording\{}\{}\{}\{}".format(rec[:5], 
                                                        rec[:-3], 
                                                        rec[-2:],
                                                        'nonrigid_reg_geco')
    
    p_suite2p = Path(p_rec+r'\suite2p_anat_detec\plane0')
    suite2p_ops = np.load(p_suite2p / 'ops.npy', allow_pickle=True).item()
    mean_img_green = suite2p_ops['meanImg_chan2']
    mean_img_red = suite2p_ops['meanImg']
    
    generate_geco_dlight_masks_wrap(
                        mean_img_red, mean_img_green,
                        p_masks
                        )
    
    F = np.load(p_suite2p / 'F.npy')
    Fneu = np.load(p_suite2p / 'Fneu.npy')
    suite2p_stat = np.load(p_suite2p / 'stat.npy', allow_pickle=True)
    F_corr = F-0.7*Fneu
    is_soma, is_active, is_active_soma = select_geco_rois(mean_img_red, F_corr, suite2p_stat, path_result=p_masks,
                          thresholds=classification_thresholds)
    np.savez_compressed(
        p_masks/r'soma_class.npz',
        is_soma=is_soma,
        is_active=is_active,
        is_active_soma=is_active_soma,
    )
    
    test_fig_dir = Path(r"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight\TEST_PLOTS\GECO_active_soma_selection_validation")
    plot_geco_selection(F_corr, mean_img_red, suite2p_stat,
                            is_soma, is_active, is_active_soma,
                            out_dir_fig=test_fig_dir/f'{rec}_geco_soma_selection.png')
    plot_random_dff_traces(gaussian_filter1d(F_corr[is_active_soma], 1),
                           title='active_soma_traces',
                           out_path_fig=test_fig_dir/f'{rec}_active_soma_traces.png')
    plot_random_dff_traces(gaussian_filter1d(F_corr[(is_soma)&(~(is_active))], 1), 
                           title='inactive_soma_traces',
                           out_path_fig=test_fig_dir/f'{rec}_inactive_soma_traces.png')

    

def select_gcamp_rois(mean_img, F_corr, stat_array, path_result='',
                      thresholds=classification_thresholds):
    # mean_img: reference mean image of GCaMP
    # F_all: np array, (n_rois, n_frames) F_raw-0.7*Fneu corrected F trace fo all rois
    path_result = Path(path_result)
    mean_img_clip = np.clip(mean_img,
                            a_min = np.percentile(mean_img, 5),
                            a_max = np.percentile(mean_img, 98))
    base_mask, global_soma_mask = utl.generate_adaptive_membrane_mask(mean_img_clip,
                                  uniformity_threshold=0.27,
                                  min_region_size=100,
                                  hole_size_threshold=100,
                                  output_dir= path_result/'Adaptive_geco_Mask_Generation.png')
    if path_result:
        path_result = Path(path_result)
        np.save(path_result / 'global_gcamp_soma_mask.npy', global_soma_mask)
    
    # anatomical features
    roi_npix    = np.array([r['npix'] for r in stat_array])
    hollowness_scores = utl.calculate_rois_hollowness(stat_array=stat_array, soma_mask=global_soma_mask)
    compactness, area, aspect_ratio_all, _ = utl.calculate_rois_compactness(stat_array=stat_array)
    # activity
    skewness   = skew(F_corr, axis=-1)
    all_F_mean = np.nanmean(F_corr, axis=-1) # (n_rois, )
    all_F_std  = np.nanstd(F_corr, axis=-1)  # (n_rois, )
    act_thresh = all_F_mean+2*all_F_std
    conti_act_frames = run_of_ones(F_corr > act_thresh[:, None])
    conti_act_frames_max = np.array([np.max(x) if len(x)>0 else np.nan for x in conti_act_frames])
    # correlation = pearsonr(F_raw, F_neu, axis=-1)[0]   

    is_soma = (
        (roi_npix > thresholds['min_soma_roi_npix']) &
        (roi_npix < thresholds['max_soma_roi_npix']) &
        (area < thresholds['area_threshold']) &
        (compactness > thresholds['compactness_threshold_min']) & # Corrected logic for compactness
        (aspect_ratio_all < thresholds['aspect_ratio_max'])&
        (hollowness_scores > thresholds['hollowness_threshold_min'])
    )
    is_active = ((skewness > thresholds['skewness_threshold'])&
                 (conti_act_frames_max > thresholds['min_act_frames'])&
                 (conti_act_frames_max < thresholds['max_act_frames']))
    is_active_soma = (is_soma)&(is_active)
    
    
    return is_soma, is_active, is_active_soma

# Helper function to overlay grid lines
def overlay_grid(ax, shape, grid_size):
    h, w = shape
    for i in range(0, h, grid_size):
        ax.axhline(i, color='grey', linewidth=0.5, alpha=0.3)
    for j in range(0, w, grid_size):
        ax.axvline(j, color='grey', linewidth=0.5, alpha=0.3)    
    
#%% test run
if __name__ == "__main__":
    import shutil
    from tqdm import tqdm
    from common.plotting_functions_Jingyu import plot_random_dff_traces
    exp = 'Dbh_dlight'
    # exp = 'geco_dlight'
    
    if exp == 'Dbh_dlight':
        
        from dlight_imaging.Dbh_dlight.recording_list import rec_lst_dlight_dbh as rec_lst
        # exp = 'dlight_Ai14_Dbh'
        # f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_new.pkl".format(exp)
        # df_selected = pd.read_pickle(f_out_df_selected)
        # rec_lst = df_selected.index.tolist()
        # ex_lst = [
        # 'AC963-20250218-04',
        # 'AC964-20250203-04',
        # 'AC964-20250205-04',
        # 'AC964-20250218-02',
        # 'AC966-20250217-04',
        # 'AC966-20250219-02',
        # ]
        
        # rec_lst_tmp = [i for i in rec_lst if i not in ex_lst]
        rec_lst_tmp = rec_lst[:1]
        # dilation_steps = (0, 2, 4, 6, 8, 10)
        DILATION_STEPS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        # dilation_steps = (0, )
        for rec in tqdm(rec_lst):
            print(f'\nProcessing {rec}...')
            p_mask_result = Path(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res_grid_free_dilation\{rec}\masks")
            # shutil.rmtree(p_mask_result)
            if not p_mask_result.exists():
                p_mask_result.mkdir(parents=True)
            shutil.copy(rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res\{rec}\masks\ch2_FOV.npy_ROI_dict_selected.npy", 
                        rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\Dbh_dlight\regression_res_grid_free_dilation\{rec}\masks")     
            extract_axon_dlight_masks(rec, p_mask_result, DILATION_STEPS,
                                      constrain_to_grid=False,
                                      visualize_neu = True,
                                      visualize_dilation = True,
                                      visualize_regressor = True)
            ## plot neuropil example
            # fiber_neuropil = np.load(p_mask_result/'fiber_neuropil_masks_dilation_k=0.npy')
            # fiber_mask = np.load(p_mask_result/'dilated_global_axon_k=0.npy')
            # fiber_neuropil_valid = fiber_neuropil[np.sum(fiber_neuropil, axis=(-2, -1))>0]
            # fiber_neuropil_mask = fiber_neuropil_valid.copy()
            # fig, ax = plt.subplots(dpi=300)
            # ax.imshow(fiber_mask, cmap='Reds', alpha=.3)
            # fiber_neuropil_mask = fiber_neuropil_valid[98]
            # fiber_neuropil_mask = np.where(fiber_neuropil_mask<1, np.nan, 1)
            # ax.imshow(fiber_neuropil_mask, cmap='tab20', alpha=.3)
            # overlay_grid(ax, fiber_mask.shape, grid_size=16)
            # plt.show()
    elif exp == 'geco_dlight':
        exp = r'dlight_GECO_Ai14_Dbh'
        f_out_df_selected = r"Z:\Jingyu\Code\dlight_imgaing\{}\df_behaviour_info_selected_corr.pkl".format(exp)
        df_selected = pd.read_pickle(f_out_df_selected)
        df_selected = df_selected.loc[(df_selected['speed_corr_single_trial_r_median']<0.3)
                                      # &(~df_selected.index.str.contains('AC991'))
                                      ]
        rec_lst = df_selected.index.tolist()
        
        for rec in rec_lst:
            print(f'\nProcessing {rec}...')
            p_mask_result = rf"Z:\Jingyu\LC_HPC_manuscript\raw_data\geco_dlight\regression_res\{rec}\masks"
            if not os.path.exists(p_mask_result):
                os.makedirs(p_mask_result)
            extract_geco_dlight_masks(rec, p_mask_result)
            
        
    
   