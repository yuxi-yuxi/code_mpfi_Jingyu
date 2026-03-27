# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 00:06:29 2026

@author: Jingyu Cao
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from common.plotting_functions_Jingyu import save_fig

def extendROI(ypix, xpix, Ly, Lx, niter=1):
    """ extend ypix and xpix by niter pixel(s) on each side """
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix - 1, ypix + 1), (xpix, xpix + 1, xpix - 1, xpix,
                                                       xpix))
        yx = np.array(yx)
        yx = yx.reshape((2, -1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix

def create_neuropil_masks(ypixs, xpixs, cell_pix, inner_neuropil_radius,
                          min_neuropil_pixels, circular=False):
    """ creates surround neuropil masks for ROIs in stat by EXTENDING ROI (slower if circular)

    Parameters
    ----------

    cellpix : 2D array
        1 if ROI exists in pixel, 0 if not;
        pixels ignored for neuropil computation

    Returns
    -------

    neuropil_masks : list
        each element is array of pixels in mask in (Ly*Lx) coordinates

    """
    valid_pixels = lambda cell_pix, ypix, xpix: cell_pix[ypix, xpix] < .5
    extend_by = 5

    Ly, Lx = cell_pix.shape
    assert len(xpixs) == len(ypixs)
    neuropil_ipix = []
    idx = 0
    for ypix, xpix in zip(ypixs, xpixs):
        neuropil_mask = np.zeros((Ly, Lx), bool)
        # extend to get ring of dis-allowed pixels
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx, niter=inner_neuropil_radius)
        nring = np.sum(valid_pixels(cell_pix, ypix,
                                    xpix))  # count how many pixels are valid

        nreps = count()
        ypix1, xpix1 = ypix.copy(), xpix.copy()
        while next(nreps) < 100 and np.sum(valid_pixels(
                cell_pix, ypix1, xpix1)) - nring <= min_neuropil_pixels:
            if circular:
                ypix1, xpix1 = extendROI(ypix1, xpix1, Ly, Lx,
                                         extend_by)  # keep extending
            else:
                ypix1, xpix1 = np.meshgrid(
                    np.arange(max(0,
                                  ypix1.min() - extend_by),
                              min(Ly,
                                  ypix1.max() + extend_by + 1), 1, int),
                    np.arange(max(0,
                                  xpix1.min() - extend_by),
                              min(Lx,
                                  xpix1.max() + extend_by + 1), 1, int), indexing="ij")

        ix = valid_pixels(cell_pix, ypix1, xpix1)
        neuropil_mask[ypix1[ix], xpix1[ix]] = True
        neuropil_mask[ypix, xpix] = False
        neuropil_ipix.append(np.ravel_multi_index(np.nonzero(neuropil_mask), (Ly, Lx)))
        idx += 1

    return neuropil_ipix

def roi_map_to_neuropil_masks(roi_map, inner_neuropil_radius=3, min_neuropil_pixels=350, circular=False):
    """
    Convert roi_map [n_roi, Ly, Lx] boolean array to neuropil masks with the same format.

    Parameters
    ----------
    roi_map : ndarray
        Boolean array of shape [n_roi, Ly, Lx] where each slice is an ROI mask
    inner_neuropil_radius : int
        Number of pixels to extend ROI inward before neuropil starts
    min_neuropil_pixels : int
        Minimum number of pixels in neuropil mask
    circular : bool
        Whether to use circular extension (slower) or rectangular

    Returns
    -------
    neuropil_map : ndarray
        Boolean array of shape [n_roi, Ly, Lx] where each slice is a neuropil mask
    """
    if roi_map.ndim == 3:
        n_roi, Ly, Lx = roi_map.shape
    elif roi_map.ndim == 2: # only 1 roi
        n_roi  = 1
        Ly, Lx = roi_map.shape
        roi_map = roi_map[np.newaxis, :]

    # Extract ypixs and xpixs from roi_map
    ypixs = []
    xpixs = []
    for i in range(n_roi):
        ypix, xpix = np.nonzero(roi_map[i])
        ypixs.append(ypix)
        xpixs.append(xpix)

    # Create cell_pix: combined mask of all ROIs
    cell_pix = np.any(roi_map, axis=0).astype(np.float32)

    # Get neuropil indices
    neuropil_ipix = create_neuropil_masks(ypixs, xpixs, cell_pix,
                                          inner_neuropil_radius,
                                          min_neuropil_pixels,
                                          circular)

    # Convert back to [n_roi, Ly, Lx] boolean array
    neuropil_map = np.zeros((n_roi, Ly, Lx), dtype=bool)
    for i, ipix in enumerate(neuropil_ipix):
        ypix, xpix = np.unravel_index(ipix, (Ly, Lx))
        neuropil_map[i, ypix, xpix] = True

    return neuropil_map.squeeze()

# Helper function to overlay grid lines
def overlay_grid(ax, shape, grid_size):
    h, w = shape
    for i in range(0, h, grid_size):
        ax.axhline(i, color='white', linewidth=0.5, alpha=0.3)
    for j in range(0, w, grid_size):
        ax.axvline(j, color='white', linewidth=0.5, alpha=0.3)
        
def fiber_block_to_neuropil_masks(current_axon_mask, global_axon_mask, grid_size,
                                  constrain_to_grid = True,
                                  output_dir=None,
                                  visualize=False,
                                  fig_out=None,):
    global_axon_mask = np.any(global_axon_mask , axis=(0, 1))
    h, w = global_axon_mask.shape
    n_y = (h + grid_size - 1) // grid_size
    n_x = (w + grid_size - 1) // grid_size
    neuropil_masks = np.zeros((n_y, n_x, h, w), dtype=bool)
    n = 0
    for by, ii in enumerate(range(0, h, grid_size)):          # by = block index (y)
        for bx, jj in enumerate(range(0, w, grid_size)):      # bx = block index (x)
    
            grid_map = np.zeros_like(global_axon_mask, dtype=bool)
            grid_map[ii:min(ii+grid_size, h), jj:min(jj+grid_size, w)] = True
    
            # fiber_map = current_axon_mask & grid_map
            fiber_map = current_axon_mask[by, bx, :, :]
    
            if fiber_map.any():
                fiber_neuropil = roi_map_to_neuropil_masks(fiber_map)
                fiber_neuropil = fiber_neuropil&(~fiber_map)&(~global_axon_mask)
                n += 1
                if visualize & (n<2):
                    y0 = max(ii - grid_size, 0)
                    y1 = min(ii + 2 * grid_size, h)
                    x0 = max(jj - grid_size, 0)
                    x1 = min(jj + 2 * grid_size, w)

                    fig, ax = plt.subplots(figsize=(3,3), dpi=100)
                    ax.imshow(np.where(fiber_map[y0:y1, x0:x1]>0, 1 , np.nan),
                              interpolation='none',
                              cmap='Set1', alpha=0.5)
                    ax.imshow(np.where(fiber_neuropil[y0:y1, x0:x1]>0, 1 , np.nan),
                              interpolation='none',
                              cmap='tab20b', alpha=0.5)
                    ax.set_title(f'Grid ({by}, {bx})')
                    if fig_out is not None:
                        # fig_out.mkdir(parents=True, exexist_ok=True)
                        # save dilation review
                        save_fig(fig, fig_out, 'grid_free_neuropil_review', save=1, forms=['png',])
                    
            else:
                fiber_neuropil = np.zeros_like(global_axon_mask, dtype=bool)
                
            neuropil_masks[by, bx, :, :] = fiber_neuropil
    if output_dir is None:
        return neuropil_masks
    else:
        np.save(output_dir, neuropil_masks)
        
    
        
        
    
#%%
if __name__ == '__main__':
    f_path = r"Z:\Jingyu\2P_Recording\AC989\AC989-20250709\concat\A_master.nc"
    A_master = xr.open_dataarray(f_path)
    A_master = A_master.rename({"master_uid": "unit_id"})
    roi_map = np.array(A_master).squeeze().astype(bool)
    
    # Create neuropil masks with the same format as roi_map
    neuropil_map = roi_map_to_neuropil_masks(roi_map,
                                             inner_neuropil_radius=3,
                                             min_neuropil_pixels=350)
    # validation plot
    p_suite2p = r'Z:\Jingyu\2P_Recording\AC989\AC989-20250709\02\suite2p_func_detec\plane0'
    suite2p_stat = np.load(p_suite2p+r'\stat.npy', allow_pickle=True)
    roi_stat = suite2p_stat[20]
    cell_pix = np.zeros((512, 512), dtype=bool)
    cell_pix[roi_stat['ypix'], roi_stat['xpix']] = True
    Ly, Lx = 512, 512
    neuropil_ipix = roi_stat['neuropil_mask']
    neuropil_mask_2d = np.zeros((Ly, Lx), dtype=bool)
    ypix, xpix = np.unravel_index(neuropil_ipix.astype(int), (Ly, Lx))
    neuropil_mask_2d[ypix, xpix] = True
    
    plt.imshow(cell_pix)
    plt.show()
    plt.imshow(neuropil_mask_2d)
    plt.show()
    
    
    roi = 99
    plt.imshow(roi_map[roi])
    plt.show()
    plt.imshow(neuropil_map[roi])
    plt.show()

    # Save neuropil_map as xarray DataArray with the same format as A_master
    # Copy A_master structure and replace data with neuropil_map
    A_neuropil = A_master.copy()
    A_neuropil.values = neuropil_map.reshape(A_master.shape)
    # Save to netCDF file in the same directory as A_master
    import os
    save_dir = os.path.dirname(f_path)
    save_path = os.path.join(save_dir, "A_master_neuropil.nc")
    A_neuropil.to_netcdf(save_path)
    print(f"Saved neuropil masks to: {save_path}")