# -*- coding: utf-8 -*-


"""
Created on Fri Aug 25 14:49:36 2017

@author: agiovann
"""
import cv2
from dask.dataframe.core import idxmaxmin_agg

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print(1)
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not launched under iPython')

import caiman as cm
import numpy as np
import os
import time
import pylab as pl
import scipy
import sys
import copy
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.estimates import Estimates, compare_components
from caiman.cluster import setup_cluster
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction.cnmf.cnmf import load_CNMF

# %%  ANALYSIS MODE AND PARAMETERS
preprocessing_from_scratch = False
plot_on = False
save_grid = False


try:
    if 'pydevconsole' in sys.argv[0]:
        raise Exception()
    ID = sys.argv[1]
    ID = str(np.int(ID) - 1)
    print('Processing ID:' + str(ID))
    ID = [np.int(ID)]

except:
    ID = np.arange(0,9)
    print('ID NOT PASSED')



backend_patch = 'local'
backend_refine = 'local'
n_processes = 24
base_folder = '/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/'

SNRs_grid = [1.75, 2, 2.25, 2.5]
r_val_grid = [0.75, 0.8, 0.85]
max_class_prob_rej_grid = np.array([0.05, 0.1, 0.15])
thresh_CNN_grid = [0.9, .95 ,0.99, 1]
block_size = 10000
num_blocks_per_run = 12
n_pixels_per_process = 4000



# %%
global_params = {'SNR_lowest': 0.5,
                 'min_SNR': 2,  # minimum SNR when considering adding a new neuron
                 'gnb': 2,  # number of background components
                 'rval_thr': 0.8,  # spatial correlation threshold
                 'min_rval_thr_rejected': -1.1,
                 'min_cnn_thresh': 0.99,
                 'max_classifier_probability_rejected': 0.1,
                 'p': 1,
                 'max_fitness_delta_accepted': -20,
                 'Npeaks': 5,
                 'min_SNR_patch': -10,
                 'min_r_val_thr_patch': 0.5,
                 'fitness_delta_min_patch': -5,
                 'update_background_components': True,
                 # whether to update the background components in the spatial phase
                 'low_rank_background': True,
                 # whether to update the using a low rank approximation. In the False case all the nonzero elements of the background components are updated using hals
                 # (to be used with one background per patch)
                 'only_init_patch': True,
                 'is_dendrites': False,  # if dendritic. In this case you need to set init_method to sparse_nmf
                 'alpha_snmf': None,
                 'init_method': 'greedy_roi',
                 'filter_after_patch': False,
                 'tsub': 2,
                 'ssub': 2
                 }
# %%
params_movies = []
# %%
params_movie = {'fname': 'N.03.00.t/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                'gtname': 'N.03.00.t/joined_consensus_active_regions.npy',
                # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 25,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 4,  # number of components per patch
                'gSig': [8, 8],  # expected half size of neurons
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 0,
                'fr': 7,
                'decay_time': 0.4,
                }
params_movies.append(params_movie.copy())
# %%
params_movie = {'fname': 'N.04.00.t/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                'gtname': 'N.04.00.t/joined_consensus_active_regions.npy',
                # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 5,  # number of components per patch
                'gSig': [5, 5],  # expected half size of neurons
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 0,
                'fr': 8,
                'decay_time': 1.4,  # rough length of a transient
                }
params_movies.append(params_movie.copy())

# %% yuste
params_movie = {'fname': 'YST/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                'gtname': 'YST/joined_consensus_active_regions.npy',
                # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 15,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 10,  # number of components per patch
                'gSig': [5, 5],  # expected half size of neurons
                'fr': 10,
                'decay_time': 0.75,
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 0
                }
params_movies.append(params_movie.copy())
# %% neurofinder 00.00
params_movie = {'fname': 'N.00.00/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                'gtname': 'N.00.00/joined_consensus_active_regions.npy',
                'merge_thresh': 0.8,  # merging threshold, max correlation allow
                'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 7,  # number of components per patch
                'gSig': [6, 6],  # expected half size of neurons
                'decay_time': 0.7,
                'fr': 8,
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 10
                }
params_movies.append(params_movie.copy())
# %% neurofinder 01.01
params_movie = {'fname': 'N.01.01/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                'gtname': 'N.01.01/joined_consensus_active_regions.npy',
                'merge_thresh': 0.9,  # merging threshold, max correlation allow
                'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
                'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
                'K': 7,  # number of components per patch
                'gSig': [6, 6],  # expected half size of neurons
                'decay_time': 0.4,
                'fr': 8,
                'n_chunks': 10,
                'swap_dim': False,
                'crop_pix': 2,
                }
params_movies.append(params_movie.copy())
# %% neurofinder 02.00
params_movie = {
    # 'fname': '/opt/local/Data/labeling/neurofinder.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'fname': 'N.02.00/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
    'gtname': 'N.02.00/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 6,  # number of components per patch
    'gSig': [5, 5],  # expected half size of neurons
    'fr': 30,  # imaging rate in Hz
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 10,
    'decay_time': 0.3,
}
params_movies.append(params_movie.copy())
# %% Sue Ann k53
params_movie = {  # 'fname': '/opt/local/Data/labeling/k53_20160530/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    'fname': 'K53/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
    'gtname': 'K53/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 10,  # number of components per patch
    'gSig': [6, 6],  # expected half size of neurons
    'fr': 30,
    'decay_time': 0.3,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,
}
params_movies.append(params_movie.copy())
# %% J115
params_movie = {
    # 'fname': '/opt/local/Data/labeling/J115_2015-12-09_L01_ELS/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    'fname': 'J115/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
    'gtname': 'J115/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 20,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 10,  # amounpl.it of overlap between the patches in pixels
    'K': 8,  # number of components per patch
    'gSig': [7, 7],  # expected half size of neurons
    'fr': 30,
    'decay_time': 0.4,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,

}
params_movies.append(params_movie.copy())
# %% J123
params_movie = {
    # 'fname': '/opt/local/Data/labeling/J123_2015-11-20_L01_0/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    'fname': 'J123/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
    'gtname': 'J123/joined_consensus_active_regions.npy',
    # order of the autoregressive system
    'merge_thresh': 0.8,  # merging threshold, max correlation allow
    'rf': 40,  # half-size of the patches in pixels. rf=25, patches are 50x50    20
    'stride_cnmf': 20,  # amounpl.it of overlap between the patches in pixels
    'K': 11,  # number of components per patch
    'gSig': [8, 8],  # expected half size of neurons
    'decay_time': 0.5,
    'fr': 30,
    'n_chunks': 10,
    'swap_dim': False,
    'crop_pix': 2,
}
params_movies.append(params_movie.copy())




max_counter = len(SNRs_grid)*len(r_val_grid)*len(max_class_prob_rej_grid)*len(thresh_CNN_grid)*len(params_movies)
if preprocessing_from_scratch:
    all_perfs = []
    all_rvalues = []
    all_comp_SNR_raw = []
    all_comp_SNR_delta = []
    all_predictions = []
    all_labels = []
    all_results = dict()
    ALL_CCs = []

    for params_movie in np.array(params_movies)[ID]:
        #    params_movie['gnb'] = 3
        params_display = {
            'downsample_ratio': .2,
            'thr_plot': 0.8
        }

        fname_new = os.path.join(base_folder, params_movie['fname'])
        print(fname_new)
        # %% LOAD MEMMAP FILE
        Yr, dims, T = cm.load_memmap(fname_new)
        d1, d2 = dims
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        # TODO: needinfo
        Y = np.reshape(Yr, dims + (T,), order='F')
        m_images = cm.movie(images)
        if plot_on:
            if m_images.shape[0] < 5000:
                Cn = m_images.local_correlations(swap_dim=params_movie['swap_dim'], frames_per_chunk=1500)
                Cn[np.isnan(Cn)] = 0
            else:
                Cn = np.array(cm.load(
                    ('/'.join(
                        params_movie['gtname'].split('/')[:-2] + ['projections', 'correlation_image.tif'])))).squeeze()

        check_nan = False
        # %% start cluster
        # TODO: show screenshot 10
        try:
            cm.stop_server()
            dview.terminate()
        except:
            print('No clusters to stop')

        c, dview, n_processes = setup_cluster(
            backend=backend_patch, n_processes=n_processes, single_thread=False)
        # %%
        params_dict = {'fnames': [fname_new],
                       'fr': params_movie['fr'],
                       'decay_time': params_movie['decay_time'],
                       'rf': params_movie['rf'],
                       'stride': params_movie['stride_cnmf'],
                       'K': params_movie['K'],
                       'gSig': params_movie['gSig'],
                       'merge_thr': params_movie['merge_thresh'],
                       'p': global_params['p'],
                       'nb': global_params['gnb'],
                       'only_init': global_params['only_init_patch'],
                       'dview': dview,
                       'method_deconvolution': 'oasis',
                       'border_pix': params_movie['crop_pix'],
                       'low_rank_background': global_params['low_rank_background'],
                       'rolling_sum': True,
                       'nb_patch': 1,
                       'check_nan': check_nan,
                       'block_size_temp': block_size,
                       'block_size_spat': block_size,
                       'num_blocks_per_run_spat': num_blocks_per_run,
                       'num_blocks_per_run_temp': num_blocks_per_run,
                       'n_pixels_per_process': n_pixels_per_process,
                       'ssub': 2,
                       'tsub': 2,
                       'p_tsub': 1,
                       'p_ssub': 1,
                       'thr_method': 'nrg'
                       }

        init_method = global_params['init_method']

        opts = params.CNMFParams(params_dict=params_dict)

        cnm = load_CNMF(fname_new[:-5] + '_cnmf_gsig.hdf5')


        # %% prepare ground truth masks
        gt_file = os.path.join(os.path.split(fname_new)[0],
                               os.path.split(fname_new)[1][:-4] + 'match_masks.npz')
        with np.load(gt_file, encoding='latin1') as ld:
            print(ld.keys())
            Cn_orig = ld['Cn']

            gt_estimate = Estimates(A=scipy.sparse.csc_matrix(ld['A_gt'][()]), b=ld['b_gt'], C=ld['C_gt'],
                                    f=ld['f_gt'], R=ld['YrA_gt'], dims=(ld['d1'], ld['d2']))

        min_size_neuro = 3 * 2 * np.pi
        max_size_neuro = (2 * params_dict['gSig'][0]) ** 2 * np.pi
        gt_estimate.threshold_spatial_components(maxthr=0.2, dview=dview)
        gt_estimate.remove_small_large_neurons(min_size_neuro, max_size_neuro)
        _ = gt_estimate.remove_duplicates(predictions=None, r_values=None, dist_thr=0.1, min_dist=10,
                                          thresh_subset=0.6)
        print(gt_estimate.A_thr.shape)

        for gr_snr in SNRs_grid:
            for grid_rval in r_val_grid:
                for grid_max_prob_rej in max_class_prob_rej_grid:
                    for grid_thresh_CNN in thresh_CNN_grid:
                        cnm2 = copy.deepcopy(cnm)
                        global_params['min_SNR'] = gr_snr
                        global_params['rval_thr'] = grid_rval
                        global_params['max_classifier_probability_rejected'] = grid_max_prob_rej
                        global_params['min_cnn_thresh'] = grid_thresh_CNN

                        # %% check quality of components and eliminate low quality
                        cnm2.params.set('quality', {'SNR_lowest': global_params['SNR_lowest'],
                                                    'min_SNR': global_params['min_SNR'],
                                                    'rval_thr': global_params['rval_thr'],
                                                    'rval_lowest': global_params['min_rval_thr_rejected'],
                                                    'use_cnn': True,
                                                    'min_cnn_thr': global_params['min_cnn_thresh'],
                                                    'cnn_lowest': global_params['max_classifier_probability_rejected'],
                                                    'gSig_range': None})

                        t1 = time.time()
                        cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
                        cnm2.estimates.select_components(use_object=True)
                        t_eva_comps = time.time() - t1
                        print(' ***** ')
                        print((len(cnm2.estimates.C)))
                        # %%
                        if plot_on:
                            cnm2.estimates.plot_contours(img=Cn)

                        # %% prepare CNMF maks
                        cnm2.estimates.threshold_spatial_components(maxthr=0.2, dview=dview)
                        cnm2.estimates.remove_small_large_neurons(min_size_neuro, max_size_neuro)
                        _ = cnm2.estimates.remove_duplicates(r_values=None, dist_thr=0.1, min_dist=10, thresh_subset=0.6)

                        # %%
                        params_display = {
                            'downsample_ratio': .2,
                            'thr_plot': 0.8
                        }

                        pl.rcParams['pdf.fonttype'] = 42
                        font = {'family': 'Arial',
                                'weight': 'regular',
                                'size': 20}
                        pl.rc('font', **font)

                        plot_results = False
                        if plot_results:
                            pl.figure(figsize=(30, 20))

                        tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = compare_components(gt_estimate, cnm2.estimates,
                                                                                                  Cn=Cn_orig, thresh_cost=.8,
                                                                                                  min_dist=10,
                                                                                                  print_assignment=False,
                                                                                                  labels=['GT', 'Offline'],
                                                                                                  plot_results=False)

                        print(fname_new+str({a: b.astype(np.float16) for a, b in performance_cons_off.items()}))
                        cnm2.estimates.A_thr = scipy.sparse.csc_matrix(cnm2.estimates.A_thr)


                        performance_cons_off['fname_new'] = fname_new
                        performance_tmp = performance_cons_off.copy()
                        performance_tmp['tp_gt'] = tp_gt
                        performance_tmp['tp_comp'] = tp_comp
                        performance_tmp['fn_gt'] = fn_gt
                        performance_tmp['fp_comp'] = fp_comp

                        ALL_CCs.append([scipy.stats.pearsonr(a, b)[0] for a, b in
                                        zip(gt_estimate.C[tp_gt], cnm2.estimates.C[tp_comp])])

                        performance_tmp['ALL_CCs'] = ALL_CCs


                        if save_grid:
                            grid_fold = os.path.join(os.path.split(fname_new)[0], 'grid')
                            os.makedirs(grid_fold, exist_ok=True)
                            grid_file = os.path.join(grid_fold, 'perf_grid_0918_' + str(gr_snr) + '_' + str(grid_rval)
                                  + '_' + str(grid_max_prob_rej) + '_' + str(grid_thresh_CNN) + '.npz')
                            print(grid_file + '__' + str({a: b.astype(np.float16) for a, b in performance_cons_off.items() if type(b) is not str}))
                            np.savez(grid_file, all_results=performance_tmp)




else:
    #%% performance grid search parameters
    grd_fld_nm = 'grid'
    records = []
    for gr_snr in SNRs_grid:
        for grid_rval in r_val_grid:
            print(grid_rval)
            for grid_max_prob_rej in max_class_prob_rej_grid:
                for grid_thresh_CNN in thresh_CNN_grid:
                    global_params['min_SNR'] = gr_snr
                    global_params['rval_thr'] = grid_rval
                    global_params['max_classifier_probability_rejected'] = grid_max_prob_rej
                    global_params['min_cnn_thresh'] = grid_thresh_CNN
                    for params_movie in np.array(params_movies)[ID]:
                        #    params_movie['gnb'] = 3
                        params_display = {
                            'downsample_ratio': .2,
                            'thr_plot': 0.8
                        }

                        fname_new = os.path.join(base_folder, params_movie['fname'])
                        grid_fold = os.path.join(os.path.split(fname_new)[0], 'grid')
                        os.makedirs(grid_fold, exist_ok=True)
                        grid_file = os.path.join(grid_fold,
                                                 'perf_grid_0918_' + str(gr_snr) + '_' + str(grid_rval)
                                                 + '_' + str(grid_max_prob_rej) + '_' + str(
                                                     grid_thresh_CNN) + '.npz')
                        if os.path.exists(grid_file):
                            with np.load(grid_file) as ld:
                                perf = ld['all_results'][()]
                                records.append([fname_new.split('/')[-2],str(round(gr_snr,2)), str(round(grid_rval,2)), str(round(grid_max_prob_rej,2)), str(round(grid_thresh_CNN,2)), perf['recall'],perf['precision'],perf['f1_score']])
                        else:
                            print('SKIPPED:' + grid_file)
                        # print(grid_fold + ' ' + str(len(perf['tp_comp']) + len('fp_comp')))
                        # print(grid_fold + ' ' + str(len(perf['tp_gt']) + len('fn_gt')))



    #%%
    if save_grid:
        np.savez('/mnt/ceph/neuro/DataForPublications/DATA_PAPER_ELIFE/ALL_RECORDS_GRID_FINAL.npz', records=records)



