# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 14:32:59 2025

@author: Jingyu Cao
@contributor: Yingxue Wang
"""

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.filters import threshold_local
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
# from skimage.measure import regionprops
import cv2
from scipy.ndimage import gaussian_filter, uniform_filter, minimum_filter, binary_dilation
from common.plotting_functions_Jingyu import save_fig

def load_axon_mask(p_masks):
    p_fiber_mask = p_masks / r'ch2_FOV.npy_ROI_dict_selected.npy'
    print(f'loading axon masks from {p_fiber_mask}...')
    if os.path.exists(p_fiber_mask):
        axon_mask = np.zeros((512, 512), dtype='bool')
        axon_mask_dict = np.load(p_fiber_mask, allow_pickle=True).item()
        for r_idx, r in axon_mask_dict.items():
            axon_mask[r['ypix'], r['xpix']]=True
    return axon_mask

def save_mask(mask, output_dir, filebase, save_tif=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save as an 8-bit TIFF (0 for False, 255 for True), common for image viewers
    if save_tif:
        tifffile.imwrite(output_dir / f'{filebase}.tiff', mask.astype('uint8') * 255)
    # Save as a NumPy binary file, preserving the boolean or integer type
    np.save(output_dir / f'{filebase}.npy', mask)

def split_map_to_grids_full(map2d, n_y, n_x):
    h, w = map2d.shape

    # grid boundaries
    y_edges = np.linspace(0, h, n_y + 1, dtype=int)
    x_edges = np.linspace(0, w, n_x + 1, dtype=int)

    out = np.zeros((n_y, n_x, h, w), dtype=map2d.dtype)

    for gy in range(n_y):
        for gx in range(n_x):
            y0, y1 = y_edges[gy], y_edges[gy + 1]
            x0, x1 = x_edges[gx], x_edges[gx + 1]
            out[gy, gx, y0:y1, x0:x1] = map2d[y0:y1, x0:x1]

    return out
       
def generate_adaptive_membrane_mask(
    mean_img,
    # --- preproc ---
    gaussian_sigma=1.5,
    # --- watershed (cells) – defaults ---
    peak_min_distance=5,
    adaptive_block_size=21, # larger & odd works better for uneven fields
    # --- region classification (cell vs neuropil) ---
    valley_radius=10,              # ~2–4 µm; controls local min/valley scale
    uniformity_thresh=3.5,            # higher -> stricter "cell-like"
    # --- neuropil local thresholding ---
    z_tau=3.8,                    # local z-score cut for neuropil
    min_region_size=150,
    hole_size_threshold=150,
    # plot process
    visualize = 1
):
    """
    Improved neuropil handling:
      - Classify pixel neighborhoods as 'cell-like' (dark somas) vs 'neuropil-like' (uniform)
        using a valley-contrast score.
      - In neuropil-like areas, keep pixels above local mean by z_tau * local std.
      - Keep your watershed membrane path for cell-like areas.

    Returns
    -------
    base_mask, final_mask, debug (dict of intermediate arrays)
    """
    img = mean_img.astype(np.float32)
    img = (img - np.percentile(img, 2)) / (np.percentile(img, 98) - np.percentile(img, 2) + 1e-6)
    img = np.clip(img, 0, 1)

    # --- base local bright mask (broad containment) ---
    img_s = gaussian_filter(img, sigma=gaussian_sigma)
    base_mask = img_s > threshold_local(img_s, block_size=adaptive_block_size, method='gaussian')

    # ========== region classification: cell-like vs neuropil-like ==========
    # fast local stats
    r = valley_radius
    # local mean / mean of squares for std
    mu = uniform_filter(img_s, size=2*r+1, mode='reflect')
    mu2 = uniform_filter(img_s**2, size=2*r+1, mode='reflect')
    sigma = np.sqrt(np.maximum(mu2 - mu**2, 0.0))

    # local minimum to capture dark somas ("valleys")
    loc_min = minimum_filter(img_s, size=2*r+1, mode="reflect")
    # valley-contrast score: big when bright rims surround a dark soma
    uniformity = mu / (sigma + 1e-6)
    neuropil_like = (uniformity > uniformity_thresh)
    cell_like = ~neuropil_like

    # ========== membranes via watershed in cell-like zones ==========
    cell_regions_mask = base_mask & cell_like
    cell_membrane_mask = cell_regions_mask
    
    # ========== neuropil via local z-threshold in neuropil-like zones ==========
    z = img_s / (sigma + 1e-6)
    # neuropil_mask = (z > z_tau) & base_mask & neuropil_like
    neuropil_mask = (z > z_tau) & neuropil_like
    
    # clean gently (no aggressive closing)
    neuropil_mask = remove_small_holes(neuropil_mask, area_threshold=hole_size_threshold)
    neuropil_mask = remove_small_objects(neuropil_mask, min_size=min_region_size)

    # ========== combine ==========
    final_mask = (cell_membrane_mask | neuropil_mask)
    final_mask = remove_small_objects(final_mask, min_size=min_region_size)

    debug = dict(img_s=img_s, mu=mu, sigma=sigma, loc_min=loc_min,
                 uniformity=uniformity, neuropil_like=neuropil_like,
                 cell_like=cell_like, z=z,
                 cell_membrane_mask=cell_membrane_mask, neuropil_mask=neuropil_mask)
    
    if visualize:
        neuropil_mask = neuropil_like.astype(bool)
        neuropil_zmap = debug['z']
        neuropil_zmap[~neuropil_mask] = np.nan
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax[0,0].imshow(img, cmap='gray'); ax[0,0].set_title('Original'); ax[0,0].axis('off')
        ax[0,1].imshow(uniformity, cmap='magma'); ax[0,1].set_title('uniformity'); ax[0,1].axis('off')
        ax[0,2].imshow(neuropil_like, cmap='coolwarm'); ax[0,2].set_title('Neuropil-like'); ax[0,2].axis('off')
        ax[1,0].imshow(cell_membrane_mask, cmap='gray'); ax[1,0].set_title('Cell membranes'); ax[1,0].axis('off')
        ax[1,1].imshow(neuropil_zmap, cmap='viridis', vmax=np.nanpercentile(neuropil_zmap, 99));ax[1,1].set_title('Neuropil_Z_map'); ax[1,1].axis('off')
        ax[1,2].imshow(img, cmap='gray'); ax[1,2].imshow(final_mask, alpha=0.35, cmap='Reds')
        ax[1,2].set_title('Final'); ax[1,2].axis('off')
        plt.tight_layout()
        # plt.show()
        return base_mask, final_mask, fig
    
    else:
        return final_mask

# Helper function to overlay grid lines
def overlay_grid(ax, shape, grid_size, color='white'):
    h, w = shape
    for i in range(0, h, grid_size):
        ax.axhline(i, color=color, linewidth=0.5, alpha=0.3)
    for j in range(0, w, grid_size):
        ax.axvline(j, color=color, linewidth=0.5, alpha=0.3)

def axon_mask_dilation(global_axon_mask, global_dlight_mask, ref_img,
                       dilation_steps,
                       method='binary',
                       output_dir=None,
                       constrain_to_grid=True,
                       grid_size=16):

    # make sure mask can be processed by cv2 dilation function
    global_axon_mask_unit8 = global_axon_mask.astype(np.uint8)

    fig, axs = plt.subplots(1, len(dilation_steps), figsize=(4*len(dilation_steps), 4), squeeze=False)
    axs = axs[0]  # flatten for easier indexing
    
    if method == 'cv2':
        for i, k_size in enumerate(dilation_steps):
            if k_size == 0:
                dilated_roi_last = global_axon_mask
                ax = axs[0]
                ax.imshow(ref_img,
                          vmin = np.percentile(ref_img, 1),
                          vmax = np.percentile(ref_img, 98),
                          cmap='gray')
                ax.imshow(np.where(global_axon_mask>0, 1 , np.nan),
                          interpolation='none',
                          cmap='Set1', alpha=0.5)
                if constrain_to_grid:
                    overlay_grid(ax, global_axon_mask.shape, grid_size)
                ax.set(title='original')
                ax.axis("off")
                save_mask(global_axon_mask, output_dir, filebase=f'dilated_global_axon_k={k_size}')
                # save_mask((global_axon_mask & global_dlight_mask), 
                #           output_dir, filebase=f'dilated_global_axon_and_dlight_mask_k={k_size}')
                # save_mask((global_axon_mask | global_dlight_mask), 
                #           output_dir, filebase=f'dilated_global_axon_or_dlight_mask_k={k_size}')
                continue
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)) # elliptical kernel

            if constrain_to_grid:
                # Dilate each grid block independently
                h, w = global_axon_mask_unit8.shape
                dilated_roi = np.zeros_like(global_axon_mask_unit8, dtype=np.uint8)
                for ii in range(0, h, grid_size):
                    for jj in range(0, w, grid_size):
                        grid_block = global_axon_mask_unit8[ii:ii+grid_size, jj:jj+grid_size]
                        if np.any(grid_block):
                            dilated_block = cv2.dilate(grid_block, kernel, iterations=1)
                            dilated_roi[ii:ii+grid_size, jj:jj+grid_size] = dilated_block
                dilated_roi = dilated_roi.astype('bool')
            else:
                dilated_roi = cv2.dilate(global_axon_mask_unit8, kernel, iterations=1)
                dilated_roi = dilated_roi.astype('bool')

            dilated_only = dilated_roi & (~dilated_roi_last)
            dilated_only = dilated_only.astype('bool')
            dilated_roi_last = dilated_roi
            
            # save masks
            save_mask(dilated_only, output_dir, filebase=f'dilated_global_axon_k={k_size}')
            # save_mask((dilated_only & global_dlight_mask), 
            #           output_dir, filebase=f'dilated_global_axon_and_dlight_mask_k={k_size}')
            # save_mask((dilated_only | global_dlight_mask), 
            #           output_dir, filebase=f'dilated_global_axon_or_dlight_mask_k={k_size}')
            
            ax = axs[i]
            ax.imshow(ref_img,
                      vmin = np.percentile(ref_img, 1),
                      vmax = np.percentile(ref_img, 98),
                      cmap='gray')
            ax.imshow(np.where(dilated_only>0, 1 , np.nan),
                      interpolation='none',
                      cmap='Set1', alpha=0.5)
            if constrain_to_grid:
                overlay_grid(ax, ref_img.shape, grid_size)
            ax.set(title=f'k_size={k_size}')
            ax.axis("off")

    if method == 'binary':
        for i, k_size in enumerate(dilation_steps):
            if k_size == 0:
                dilated_roi_last = global_axon_mask
                ax = axs[0]
                ax.imshow(ref_img,
                          vmin = np.percentile(ref_img, 1),
                          vmax = np.percentile(ref_img, 98),
                          cmap='gray')
                ax.imshow(np.where(global_axon_mask>0, 1 , np.nan),
                          interpolation='none',
                          cmap='Set1', alpha=0.5)
                if constrain_to_grid:
                    overlay_grid(ax, global_axon_mask.shape, grid_size)
                ax.set(title='original')
                ax.axis("off")
                # save masks
                save_mask(global_axon_mask, output_dir, filebase=f'dilated_global_axon_k={k_size}')
                continue
            
            if constrain_to_grid:
                # Dilate each grid block independently
                h, w = global_axon_mask.shape
                dilated_roi = np.zeros_like(global_axon_mask, dtype=bool)
                for ii in range(0, h, grid_size):
                    for jj in range(0, w, grid_size):
                        grid_block = global_axon_mask[ii:ii+grid_size, jj:jj+grid_size]
                        if np.any(grid_block):
                            dilated_block = binary_dilation(grid_block, iterations=k_size)
                            dilated_roi[ii:ii+grid_size, jj:jj+grid_size] = dilated_block
            else:
                dilated_roi = binary_dilation(global_axon_mask, iterations=k_size)

            dilated_roi = dilated_roi.astype('bool')

            dilated_only = dilated_roi & (~dilated_roi_last)
            dilated_only = dilated_only.astype('bool')
            dilated_roi_last = dilated_roi
            
            # save masks
            save_mask(dilated_only, output_dir, filebase=f'dilated_global_axon_k={k_size}')
            
            ax = axs[i]
            ax.imshow(ref_img,
                      vmin = np.percentile(ref_img, 1),
                      vmax = np.percentile(ref_img, 98),
                      cmap='gray')
            ax.imshow(np.where(dilated_only>0, 1 , np.nan),
                      interpolation='none',
                      cmap='Set1', alpha=0.5)
            if constrain_to_grid:
                overlay_grid(ax, ref_img.shape, grid_size)
            ax.set(title=f'k_size={k_size}')
            ax.axis("off")

    fig.tight_layout(pad=0.5)
    plt.savefig(output_dir / 'axon_dilation_review.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def axon_mask_dilation_grid_free(global_axon_mask, global_dlight_mask, ref_img,
                       dilation_steps,
                       method='binary',
                       output_dir=None,
                       constrain_to_grid=False,
                       grid_size=16,
                       visualize=False,
                       fig_out=None):
    
    h, w = global_axon_mask.shape
    n_y = (h + grid_size - 1) // grid_size
    n_x = (w + grid_size - 1) // grid_size
    
    if visualize:
        fig, axs = plt.subplots(1, len(dilation_steps), 
                                figsize=(4*len(dilation_steps), 4), 
                                squeeze=False, dpi=150)
        axs = axs[0]  # flatten for easier indexing
        fig.suptitle('grid_free_dilation_review', size=7)

    if method == 'binary':
        dilated_masks_last = split_map_to_grids_full(global_axon_mask, n_y, n_x)
        for i, k_size in enumerate(dilation_steps):
            dilated_masks = np.zeros((n_y, n_x, h, w), dtype=bool)
            if k_size == 0:
                # dilated_roi_last = global_axon_mask
                # ax = axs[0]
                # ax.imshow(ref_img,
                #           vmin = np.percentile(ref_img, 1),
                #           vmax = np.percentile(ref_img, 98),
                #           cmap='gray')
                # ax.imshow(np.where(global_axon_mask>0, 1 , np.nan),
                #           interpolation='none',
                #           cmap='Set1', alpha=0.5)
                # if constrain_to_grid:
                #     overlay_grid(ax, global_axon_mask.shape, grid_size)
                # ax.set(title='original')
                # ax.axis("off")
                
                # save masks
                global_axon_mask_split = split_map_to_grids_full(global_axon_mask, n_y, n_x)
                save_mask(global_axon_mask_split, output_dir, 
                          filebase=f'dilated_global_axon_k={k_size}',
                          save_tif=False)
                continue
            # Dilate each grid block independently
            n = 0
            for by, ii in enumerate(range(0, h, grid_size)):          # by = block index (y)
                for bx, jj in enumerate(range(0, w, grid_size)):      # bx = block index (x)
                    grid_map = np.zeros_like(global_axon_mask, dtype=bool)
                    grid_map[ii:min(ii+grid_size, h), jj:min(jj+grid_size, w)] = True
                    grid_fiber_map = global_axon_mask & grid_map
                    if grid_fiber_map.any():
                        dilated_block = binary_dilation(grid_fiber_map, iterations=k_size).astype('bool')
                        dilated_block_last = dilated_masks_last[by, bx, :, :] 
                        dilated_only = (dilated_block)&(~dilated_block_last)&(~global_axon_mask)
                        dilated_masks_last[by, bx, :, :] = dilated_block
                        
                        if constrain_to_grid:
                            dilated_masks[by, bx, :, :]= dilated_only&(grid_map)
                        else:
                            dilated_masks[by, bx, :, :] = dilated_only
                            n += 1
                            if visualize and (n < 2):
                                y0 = max(ii - grid_size, 0)
                                y1 = min(ii + 2 * grid_size, h)
                                x0 = max(jj - grid_size, 0)
                                x1 = min(jj + 2 * grid_size, w)
                                
                                # fig, ax = plt.subplots(figsize=(3,3), dpi=100)
                                ax = axs[i]
                                ax.imshow(
                                    np.where(global_axon_mask[y0:y1, x0:x1], 1, np.nan),
                                    interpolation='none',
                                    cmap='Set1',
                                    alpha=0.5
                                )
                            
                                ax.imshow(
                                    np.where(dilated_only[y0:y1, x0:x1], 1, np.nan),
                                    interpolation='none',
                                    cmap='tab20b',
                                    alpha=0.25
                                )
                                
                                overlay_grid(ax, shape=global_axon_mask[y0:y1, x0:x1].shape, 
                                             grid_size=16, color='grey')
                                # outline the center grid
                                center_x = jj - x0
                                center_y = ii - y0
                                rect = Rectangle(
                                    (center_x, center_y),
                                    min(grid_size, w - jj),
                                    min(grid_size, h - ii),
                                    fill=False,
                                    edgecolor='black',
                                    linewidth=1.5
                                )
                                ax.add_patch(rect)
                            
                                ax.set_title(f'Grid ({by}, {bx}), k={k_size}')
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_aspect('equal')
                                
            # save masks
            save_mask(dilated_masks, output_dir, 
                      filebase=f'dilated_global_axon_k={k_size}',
                      save_tif=False)
        
        if visualize:
            ax = axs[0]
            ax.imshow(
                np.where(global_axon_mask[y0:y1, x0:x1], 1, np.nan),
                interpolation='none',
                cmap='Set1',
                alpha=0.5
            )
            overlay_grid(ax, shape=global_axon_mask[y0:y1, x0:x1].shape, 
                         grid_size=16, color='grey')
            # outline the center grid
            center_x = jj - x0
            center_y = ii - y0
            rect = Rectangle(
                (center_x, center_y),
                min(grid_size, w - jj),
                min(grid_size, h - ii),
                fill=False,
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(rect)
            ax.set_title(f'Grid ({by}, {bx}), k=0')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        if fig_out is not None:
            # fig_out.mkdir(parents=True, exexist_ok=True)
            # save dilation review
            save_fig(fig, fig_out, 'grid_free_dilation_review', save=1, forms=['png',])
    
def dlight_regressor_mask (p_mask, 
                        ref_img,
                        dilation_steps,
                        neu_pix=3,
                        output_dir=None,
                        constrain_to_grid=True,
                        grid_size=16):

    fig, axs = plt.subplots(1, len(dilation_steps), figsize=(4*len(dilation_steps), 4), squeeze=False)
    axs = axs[0]  # flatten for easier indexing
    global_membrane_mask = np.load(p_mask/'global_dlight_mask_enhanced.npy')
    global_axon_mask = np.load(p_mask/'dilated_global_axon_k=0.npy')
    for i, k_size in enumerate(dilation_steps):
        global_axon_mask_dilated = np.load(p_mask/f'dilated_global_axon_k={k_size}.npy')
        if constrain_to_grid:
            # Dilate each grid block independently
            h, w = global_axon_mask_dilated.shape
            dilated_roi = np.zeros_like(global_axon_mask_dilated, dtype=bool)
            for ii in range(0, h, grid_size):
                for jj in range(0, w, grid_size):
                    grid_block = global_axon_mask_dilated[ii:ii+grid_size, jj:jj+grid_size]
                    if np.any(grid_block):
                        dilated_block = binary_dilation(grid_block, iterations=neu_pix)
                        dilated_roi[ii:ii+grid_size, jj:jj+grid_size] = dilated_block
        else:
            dilated_roi = binary_dilation(global_axon_mask_dilated, iterations=k_size)
        
        neuropil_mask = ((dilated_roi&(~global_axon_mask)&(~global_axon_mask_dilated))
                         &(global_membrane_mask))
        # save masks
        save_mask(neuropil_mask, output_dir, filebase=f'dlight_regressor_fiber_dilation_k={k_size}')
        
        ax = axs[i]
        ax.imshow(ref_img,
                  vmin = np.percentile(ref_img, 1),
                  vmax = np.percentile(ref_img, 98),
                  cmap='gray')
        ax.imshow(np.where(neuropil_mask>0, 1 , np.nan),
                  interpolation='none',
                  cmap='Set1', alpha=0.5)
        if constrain_to_grid:
            overlay_grid(ax, ref_img.shape, grid_size)
        ax.set(title=f'k_size={k_size}')
        ax.axis("off")

    fig.tight_layout(pad=0.5)
    plt.savefig(output_dir / 'dlight_regressor_review.png', dpi=300, bbox_inches='tight')
    plt.close()

def dlight_regressor_mask_grid_free (p_mask, 
                        ref_img,
                        dilation_steps,
                        neu_pix=3,
                        output_dir=None,
                        grid_size=16,
                        constrain_to_grid=False,
                        visualize=False,
                        fig_out=None):
    
    global_membrane_mask = np.load(p_mask/'global_dlight_mask_enhanced.npy')
    global_axon_mask = np.load(p_mask/'dilated_global_axon_k=0.npy')
    global_axon_mask = np.any(global_axon_mask , axis=(0, 1))
    h, w = global_axon_mask.shape
    n_y = (h + grid_size - 1) // grid_size
    n_x = (w + grid_size - 1) // grid_size
    
    if visualize:
        fig, axs = plt.subplots(1, len(dilation_steps), 
                                figsize=(4*len(dilation_steps), 4), 
                                squeeze=False, dpi=150)
        axs = axs[0]  # flatten for easier indexing
        fig.suptitle('grid_free_dilation_regressor_review', size=7)
        
    for i, k_size in enumerate(dilation_steps):
        global_axon_mask_dilated = np.load(p_mask/f'dilated_global_axon_k={k_size}.npy')
        regressor_masks = np.zeros((n_y, n_x, h, w), dtype=bool)
        # if k_size==0:
        #     global_axon_mask_dilated = split_map_to_grids_full(global_axon_mask_dilated, n_y, n_x)
        
        # Dilate each grid block independently
        # dilated_roi = np.zeros_like(global_axon_mask, dtype=bool)
        n = 0
        for by, ii in enumerate(range(0, h, grid_size)):          # by = block index (y)
            for bx, jj in enumerate(range(0, w, grid_size)):      # bx = block index (x)
                grid_map = np.zeros_like(global_axon_mask, dtype=bool)
                grid_map[ii:min(ii+grid_size, h), jj:min(jj+grid_size, w)] = True
                # grid_fiber_map = global_axon_mask & grid_map
                grid_fiber_map = global_axon_mask_dilated[by, bx, :, :]
                
                if grid_fiber_map.any():
                    dilated_block = binary_dilation(grid_fiber_map, iterations=neu_pix).astype('bool')
                    # dilated_block_last = dilated_block
                    regressor_mask = ((dilated_block)&(~global_axon_mask)&(~grid_fiber_map)
                                     &(global_membrane_mask))
                    if constrain_to_grid:
                        regressor_masks[by, bx, :, :]= (regressor_mask)&(grid_fiber_map)
                    else:
                        regressor_masks[by, bx, :, :] = regressor_mask
                        n += 1
                        if visualize & (n<2):
                            y0 = max(ii - grid_size, 0)
                            y1 = min(ii + 2 * grid_size, h)
                            x0 = max(jj - grid_size, 0)
                            x1 = min(jj + 2 * grid_size, w)
                            
                            # fig, ax = plt.subplots(figsize=(3,3), dpi=100)
                            ax = axs[i]
                            
                            ax.imshow(
                                np.where(global_membrane_mask[y0:y1, x0:x1], 1, np.nan),
                                interpolation='none',
                                cmap='Dark2',
                                alpha=0.15
                            )
                            
                            ax.imshow(
                                np.where(global_axon_mask[y0:y1, x0:x1], 1, np.nan),
                                interpolation='none',
                                cmap='Set1',
                                alpha=0.5
                            )
                            
                        
                            ax.imshow(
                                np.where(regressor_mask[y0:y1, x0:x1], 1, np.nan),
                                interpolation='none',
                                cmap='tab20b',
                                alpha=0.25
                            )
                            
                            overlay_grid(ax, shape=global_axon_mask[y0:y1, x0:x1].shape, 
                                         grid_size=16, color='grey')
                            # outline the center grid
                            center_x = jj - x0
                            center_y = ii - y0
                            rect = Rectangle(
                                (center_x, center_y),
                                min(grid_size, w - jj),
                                min(grid_size, h - ii),
                                fill=False,
                                edgecolor='black',
                                linewidth=1.5
                            )
                            ax.add_patch(rect)
                        
                            ax.set_title(f'Grid ({by}, {bx}), k={k_size}')
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_aspect('equal')

        # save masks
        save_mask(regressor_masks, output_dir, 
                  filebase=f'dlight_regressor_fiber_dilation_k={k_size}',
                  save_tif=False)   
    
    if fig_out is not None:
        # save dilation review
        # fig_out.mkdir(parents=True, exexist_ok=True)
        save_fig(fig, fig_out, 'grid_free_regressor_review', save=1, forms=['png',])

def plot_membrane_mask(mean_img_green, 
                      base_mask, global_dlight_mask_enhanced,
                      path_result_fig):
    fig, axs = plt.subplots(1, 3, dpi=300)
    # fig.suptitle(f'{rec}', x=0.5, y=0.8)
    ax = axs[0]
    ax.imshow(mean_img_green,
              vmin = np.percentile(mean_img_green, 1),
              vmax = np.percentile(mean_img_green, 98),
              cmap='gray')
    ax.set(title='mean_dLight')
    ax.axis("off")
    
    ax = axs[1]
    ax.imshow(mean_img_green,
              # vmin = np.percentile(mean_img_green, 1),
              # vmax = np.percentile(mean_img_green, 98),
              cmap='gray')
    ax.imshow(np.where(base_mask>0, 1 , np.nan), 
              interpolation='none',
              cmap='Set1', alpha=0.5)
    ax.set(title='membrane_mask_original')
    ax.axis("off")
    
    ax = axs[2]
    ax.imshow(mean_img_green,
              # vmin = np.percentile(mean_img_green, 1),
              # vmax = np.percentile(mean_img_green, 98),
              cmap='gray')
    ax.imshow(np.where(global_dlight_mask_enhanced>0, 1 , np.nan), 
              interpolation='none',
              cmap='Set1', alpha=0.5)
    ax.set(title='membrane_mask_final')
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(path_result_fig )
    plt.close()

def calculate_rois_hollowness(stat_array, soma_mask):
    """
    Calculate how "hollow" each ROI is by comparing it
    to a ground-truth mask of soma locations.

    Args:
        stat_array (np.ndarray): The array of dictionaries from 'stat.npy'.
        soma_mask_ground_truth (np.ndarray): A 2D boolean mask where True indicates
                                             the location of a dark soma center.

    """
    num_rois = len(stat_array)
    hollowness_scores = np.zeros(num_rois)  
    for i, roi in enumerate(stat_array):
        # Get the pixel coordinates for the current Suite2p ROI
        ypix = roi['ypix']
        xpix = roi['xpix']
        total_pixels_in_roi = len(ypix)

        if total_pixels_in_roi == 0:
            continue

        # --- The Hollowness Test ---
        # Count how many of the ROI's pixels fall on the bright membrane mask
        # We can do this by checking the value of the membrane mask at the ROI's coordinates
        pixels_on_membrane = np.sum(soma_mask[ypix, xpix])
        
        # Calculate the hollowness ratio
        hollowness_ratio = pixels_on_membrane / total_pixels_in_roi
        hollowness_scores[i] = hollowness_ratio
                    
    return hollowness_scores
    

def calculate_rois_compactness(stat_array):
    """
    Calculate the compactness for each ROI.

    Args:
        stat_array: stat loaded from stat.npy
    """
    
    num_rois = len(stat_array)
    
    compactness_all = np.zeros(num_rois)
    area_all = np.zeros(num_rois)
    perimeter_all = np.zeros(num_rois)
    aspect_ratio_all = np.zeros(num_rois)
    
    for i, roi in enumerate(stat_array):
        # --- Calculate compactness ---
        try:
            ymin, xmin = np.min(roi['ypix']), np.min(roi['xpix'])
            ymax, xmax = np.max(roi['ypix']), np.max(roi['xpix'])
            roi_img = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            roi_img[roi['ypix'] - ymin, roi['xpix'] - xmin] = 1
            area_all[i] = (ymax - ymin)*(xmax - xmin) # max area around the ROI
            props = regionprops(roi_img)[0]
            
            perimeter = props.perimeter
            compactness = 0 if perimeter == 0 else 4 * np.pi * props.area / (perimeter**2)
            ar =  props.major_axis_length/(props.minor_axis_length+1e-9)
            compactness_all[i] = compactness
            perimeter_all[i] = perimeter
            aspect_ratio_all[i] = ar
        except:
            compactness_all[i] = np.nan
            perimeter_all[i] = np.nan
            aspect_ratio_all[i] = np.nan
            
    return compactness_all, area_all, aspect_ratio_all, perimeter_all