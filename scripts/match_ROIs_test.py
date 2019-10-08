import numpy as np
import logging
import time
import imp
import os

from scipy import sparse
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat

import h5py
import hdf5storage

import caiman as cm
from caiman.base.rois import register_ROIs

#imp.load_source("register_ROIs","../CaImAn/caiman/base/rois.py")

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.ERROR)


def compare_old_vs_new(pathMouse,sessions=None,std=None,thr_cost=0.7,pl=False):

  #pathResults_old = 'resultsCNMF_MF1_LK1.mat'
  pathResults_new = 'results_OnACID.mat'
  pathSave = 'matching_old_new.mat'
  
  nS = sessions[1]-sessions[0]+1
  
  path_new = '%sSession01/%s' % (pathMouse,pathResults_new)
  f = loadmat(path_new)
  Cn0 = f['Cn']
  
  pathResults_old = 'backup/save_final/footprints.mat'
  path_old = '%s%s' % (pathMouse,pathResults_old)
  
  
  t_start = time.time()
  for s in range(sessions[0],sessions[1]+1):
    
    t_start_s = time.time()
    print('---------- Now matching session %d ----------'%s)
    dims = (512,512)
    
    pathFigDir = '%sSession%02d/pics/' % (pathMouse,s)
    #path_old = '%sSession%02d/%s' % (pathMouse,s,pathResults_old)
    path_new = '%sSession%02d/%s' % (pathMouse,s,pathResults_new)
    svname = '%sSession%02d/%s' % (pathMouse,s,pathSave)
    
    print(path_old)
    print(path_new)
    
    
    f = h5py.File(path_old,'r')
    #A_old = sparse.csc_matrix((f['A2']['data'], f['A2']['ir'], f['A2']['jc']))
    A_old = sparse.vstack([sparse.csc_matrix((f[A]['data'], f[A]['ir'], f[A]['jc']),shape=(512,512)).transpose().reshape(512*512) if 'data' in list(f[A].keys()) else sparse.csc_matrix((1,512*512)) for A in f[f['footprints/session/ROI'][s-1][0]]['A'][0]]).T.asformat('csc')
    f.close()
    
    f = loadmat(path_new)
    A_new = f['A']#.reshape(-1,dims[0],dims[1]).transpose(2,1,0).reshape(dims[0]*dims[1],-1)
    Cn = f['Cn']
    
    #N = A_old.shape[1]
    #A_old.resize(dims[0]*dims[1],N)
    print(A_old.shape[1])
    print(A_new.shape[1])
    
    #return A_old, A_new
    [matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, _, scores, shifts] = cm.base.rois.register_ROIs(A_old, A_new, dims,
                                                                                                            template1=Cn0, template2=Cn,
                                                                                                            std=std, cr=(15,15),
                                                                                                            thresh_cost=thr_cost, max_dist=8, max_thr=0.01, plot_results=pl)
    print(performance)
    
    results = dict(matched_ROIs1=matched_ROIs1,
                   matched_ROIs2=matched_ROIs2,
                   non_matched1=non_matched1,
                   non_matched2=non_matched2,
                   performance=performance,
                   scores=scores,
                   shifts=np.array(shifts))
    if os.path.exists(svname):
      os.remove(svname)
    
    savemat(svname, results)
    
    if pl:
      if not os.path.exists(pathFigDir):
        os.mkdir(pathFigDir)
      plt.savefig('%smatching_old_vs_new2.png'%pathFigDir)
      plt.close('all')
    
    print('---------- finished matching session %d.\t time taken: %s ----------'%(s,str(time.time()-t_start_s)))
  
  print('---------- all done. Overall time taken: %s ----------'%str(time.time()-t_start))
  return
  
  
  

def match_ROIs_test(pathMouse,sessions=None,thr_cost=0.7,std=(2,2),w=1/3,OnACID=True,pl=False):
  
  print('should remove deleted ones')
  
  if OnACID:
    suffix = '_OnACID'
  else:
    suffix = ''
  
  pathMatching = '%smatching' % pathMouse
  if not os.path.exists(pathMatching):
    os.mkdir(pathMatching)
    
  if std is None:
    pathSave = '%smatching/results_matching_multi_std=0_thr=%d_w=%d%s.mat'%(pathMouse,thr_cost*100,int(w*100),suffix)
  else:
    pathSave = '%smatching/results_matching_multi_std=%d_thr=%d_w=%d%s.mat'%(pathMouse,std[0],thr_cost*100,int(w*100),suffix)
  
  print(pathSave)
  if isinstance(pathMouse,str):
    assert isinstance(sessions,tuple), 'Please provide the numbers of sessions as a tuple of start and end session to be matched'
    if OnACID:
      pathResults = 'results_OnACID.mat'
    else:
      pathResults = 'resultsCNMF_MF1_LK1.mat'
    path = [('%sSession%02d/%s' % (pathMouse,i,pathResults)) for i in range(sessions[0],sessions[1]+1)]
    
  nS = len(path)
  
  
  A = [[]]*nS
  if not OnACID:
    pathResults = 'results_OnACID.mat'
    path = '%sSession01/%s' % (pathMouse,pathResults)
    f = loadmat(path)
    Cn = [f['Cn']]
    
    pathResults_old = 'backup/save_final/footprints.mat'
    path = '%s%s' % (pathMouse,pathResults_old)
    print("bla")
    print(path)
    f = h5py.File(path)
    for s in range(nS):
      A[s] = sparse.vstack([sparse.csc_matrix((f[A]['data'], f[A]['ir'], f[A]['jc']),shape=(512,512)).transpose().reshape(512*512) if 'data' in list(f[A].keys()) else sparse.csc_matrix((1,512*512)) for A in f[f['footprints/session/ROI'][s][0]]['A'][0]]).T.asformat('csc')
    
    f.close()
  else:
    Cn = [[]]*nS
    
  for s in range(nS):
    #print(path[s])
    if OnACID:
      if not os.path.exists(path[s]):
        print("File %s does not exist. Skipping..."%path[s])
        continue
      
      f = loadmat(path[s])
      A[s] = f['A']
      Cn[s] = f['Cn']
      
    #else:
      #f = h5py.File(path[s],'r')
      #Cn[s] = f['Cn'].value.transpose()
      
      #A[s] = sparse.csc_matrix((f['A2']['data'], f['A2']['ir'], f['A2']['jc']),shape=(np.prod(Cn[s].shape),f['C2'].shape[-1]))
      #f.close()
    
    A[s] = A[s].astype(np.float32)
    
    print('# ROIs: '+str(A[s].shape[1]))
  
  print("Start matching")
  t_start = time.time()
  if nS == 2:
    [matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, _, scores, shifts] = cm.base.rois.register_ROIs(A[0], A[1], Cn[0].shape, template1=Cn[0], template2=Cn[1],
                                                                                                                            std=std, cr=(15,15), max_dist=8, thresh_cost=thr_cost, 
                                                                                                                            plot_results=pl,use_opt_flow=False)
    print(performance)
    print("Time taken: %s" % str(time.time()-t_start))
    
    pathSave = pathMouse + 'matching/results_matching_single.mat'
    results = dict(matched_ROIs1=matched_ROIs1,
                   matched_ROIs2=matched_ROIs2,
                   non_matched1=non_matched1,
                   non_matched2=non_matched2,
                   performance=performance,
                   scores=scores,
                   shifts=np.array(shifts))
    
    savemat(pathSave, results)
    pl.savefig("/media/wollex/Analyze_AS3/Data/879/Figures/ROIs_match.png")
    
    return matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, scores, shifts
  else:
    [A_union, assignments, matchings, scores, shifts] = cm.base.rois.register_multisession(A, Cn[0].shape, templates=Cn,
                                                                                           thresh_cost=thr_cost, max_dist=8, max_thr=0.01, std=std,
                                                                                           plot_results=pl)
    print("Time taken: %s" % str(time.time()-t_start))
    
    #pathSave = '%smatching/results_matching_multi_std=%d_thr=%d_w=%d.mat'%(pathMouse,std[0],thr_cost*100,int(w*100))
    results = dict(A_union=sparse.csc_matrix(A_union),
                   assignments=assignments,
                   matchings=matchings,
                   scores=scores,
                   shifts=shifts)
    savemat(pathSave, results)
    
    return A_union, assignments, matchings, scores, shifts
  
  
  

#_ = match_ROIs_test("/media/wollex/Analyze_AS3/Data/34/",sessions=(1,22),std=None,thr_cost=0.7,w=1/3,OnACID=True,pl=False);
#_ = match_ROIs_test("/media/wollex/Analyze_AS3/Data/35/",sessions=(1,22),std=None,thr_cost=0.7,w=1/3,OnACID=True,pl=False);
#_ = match_ROIs_test("/media/wollex/Analyze_AS3/Data/65/",sessions=(1,44),std=None,thr_cost=0.7,w=1/3,OnACID=True,pl=False);
#_ = match_ROIs_test("/media/wollex/Analyze_AS3/Data/66/",sessions=(1,45),std=None,thr_cost=0.7,w=1/3,OnACID=True,pl=False);
#_ = match_ROIs_test("/media/wollex/Analyze_AS3/Data/72/",sessions=(1,44),std=None,thr_cost=0.7,w=1/3,OnACID=True,pl=False);
#_ = match_ROIs_test("/media/wollex/Analyze_AS3/Data/243/",sessions=(1,71),std=None,thr_cost=0.7,w=1/3,OnACID=True,pl=False);
#_ = match_ROIs_test("/media/wollex/Analyze_AS3/Data/244/",sessions=(1,44),std=None,thr_cost=0.7,w=1/3,OnACID=True,pl=False);