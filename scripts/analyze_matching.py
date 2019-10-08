
import random, cv2
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as pl

def analyze_matching(pathMouse,fileName,sessions=None,idx=None,sc_thr=None):
  
  pl.close('all')
  pathName = '%smatching/%s' % (pathMouse,fileName)
  print(pathName)
  f = loadmat(pathName)
  
  
  shifts = np.array(f['shifts'])
  print(shifts)
  d_shifts = np.array([np.sqrt(x**2+y**2) for x,y in shifts])
  high_sc = np.where(d_shifts < 1)
  print(high_sc[0])
  print(high_sc[0].shape)
  
  
  ## choose ROI to analyze
  if idx is None:
    #idx = np.random.randint(f['matched_ROIs1'].shape[1])
    #idx_tmp = np.random.randint(high_sc.shape[0])
    idx = random.choice(high_sc[0])
  print(idx)
  ROI1 = f['matched_ROIs1'][0][idx]
  ROI2 = f['matched_ROIs2'][0][idx]
  print(ROI1)
  print(ROI2)
  print('now analyzing ROIs %d & %d' % (ROI1,ROI2))
  
  ## calculate / print some statistics (shift, different C-values)
  print("optimal shift = %5.3g"%d_shifts[idx])
  print(shifts[idx])
  
  print("spatial correlation: \t best = %5.3g, \t unshifted = %5.3g, \t discounted = %5.3g" % (f['scores'][0][0][ROI1,ROI2],f['scores'][1][0][ROI1,ROI2],f['scores'][2][0][ROI1,ROI2]))
  
  ## and display
  pathSession1 = '%sSession%02d/results_OnACID.mat' % (pathMouse,sessions[0])
  ROIs1_ld = loadmat(pathSession1)
  
  pathSession2 = '%sSession%02d/results_OnACID.mat' % (pathMouse,sessions[1])
  ROIs2_ld = loadmat(pathSession2)
  
  print(pathSession1)
  print(pathSession2)
  
  Cn = ROIs1_ld['Cn']
  Cn2 = ROIs2_ld['Cn']
  Cn -= Cn.min()
  Cn /= Cn.max()
  Cn2 -= Cn2.min()
  Cn2 /= Cn2.max()
  dims = Cn.shape
  
  x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32))
  
  Cn_norm = np.uint8(Cn*(Cn > 0)*255)
  Cn2_norm = np.uint8(Cn2*(Cn2 > 0)*255)
  flow = cv2.calcOpticalFlowFarneback(np.uint8(Cn_norm*255),
                                      np.uint8(Cn2_norm*255),
                                      None,0.5,3,128,3,7,1.5,0)
  x_remap = (flow[:,:,0] + x_grid).astype(np.float32) 
  y_remap = (flow[:,:,1] + y_grid).astype(np.float32)
  
  X = np.arange(0,512,1)
  Y = np.arange(0,512,1)
  
  X,Y = np.meshgrid(X,Y)
  
  #print(x_remap)
  #print(y_remap)
  
  U = x_remap - X
  V = y_remap - Y
  
  idxes = 15
  fig, ax = pl.subplots(figsize=(10,10))
  q = ax.quiver(X[::idxes,::idxes], Y[::idxes,::idxes], U[::idxes,::idxes], V[::idxes,::idxes], angles='xy', scale_units='xy', scale=1, headwidth=4,headlength=4, width=0.002, units='width')
  ax.set_xlim([0,dims[0]])
  ax.set_ylim([0,dims[1]])
  pl.show(block=False)
  
  C = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(Cn) * np.fft.fft2(np.rot90(Cn2,2)))))
  C_max = np.max(C)
  max_pos = np.where(C==np.max(C))
  x_shift = (max_pos[0] - (dims[0]/2+1).astype(int)
  y_shift = (max_pos[1] - (dims[1]/2+1).astype(int)
  
  print("shift of templates: (%4.2g,%4.2g)" % (x_shift[0],y_shift[0]))
  
  level = 0.1
  cmap = 'viridis'
  
  A1 = ROIs1_ld['A'][:,ROI1]
  A2 = ROIs2_ld['A'][:,ROI2]
  A1 = np.array(A1.reshape(dims).todense())
  A2 = np.array(A2.reshape(dims).todense()).astype(np.float32)
  
  #print(type(A2))
  #print(A2.dtype)
  
  A2 = cv2.remap(A2, x_remap, y_remap, cv2.INTER_NEAREST)
  
  
  #A1 /= np.max(A1)
  #A2 /= np.max(A2)
  
  #pl.figure()
  #lp, hp = np.nanpercentile(Cn, [5, 95])
  #pl.imshow(Cn, vmin=lp, vmax=hp, cmap=cmap)
  #pl.contour(A1, levels=[level], colors='w', linewidths=1)
  #pl.contour(A2, levels=[level], colors='r', linewidths=1)
  #pl.show(block=False)