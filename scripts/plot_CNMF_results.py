from scipy.io import loadmat
import scipy
import matplotlib.pyplot as pl
import numpy as np

def plot_CNMF_results(pathMouse,session,extent):
  
  f = loadmat(pathMouse + '/matching/results_matching_multi_std=0_thr=70_w=33_OnACID.mat')
  assignments = f['assignments']
  
  path = pathMouse + 'Session%02d/results_OnACID.mat'%session[0]
  f = loadmat(path);
  Cn0 = f['Cn']
  
  for s in range(session[0],session[1]+1):
    path = pathMouse + 'Session%02d/results_OnACID.mat'%s
    f = loadmat(path);
    
    dims = f['Cn'].shape
    d1,d2 = dims
    A = f['A']
    
    if 'ndarray' not in str(type(extent)):
      extent = np.array(extent)
    
    Coor = np.matrix([np.outer(np.ones(d2), np.arange(d1)).ravel(),
                            np.outer(np.arange(d2), np.ones(d1)).ravel()], dtype=A.dtype)
    Anorm = scipy.sparse.vstack([a/a.sum() for a in A.T]).T
    cm = np.array((Coor * Anorm).T)
    
    cmap='viridis';
    
    level = 0.98
    fig = pl.figure(figsize=(2.5,2.5),frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    pl.rcParams['pdf.fonttype'] = 42
    font = {'family': 'Myriad Pro',
            'weight': 'regular',
            'size': 10}
    pl.rc('font', **font)
    
    lp, hp = np.nanpercentile(f['Cn'], [5, 95])
    ax.imshow(f['Cn'], vmin=lp, vmax=hp, cmap=cmap)
    
    C = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(Cn0) * np.fft.fft2(np.rot90(f['Cn'],2)))))
            
    max_pos = np.where(C==np.max(C))
    x_shift = (max_pos[0] - (dims[0]/2-1)).astype(int)
    y_shift = (max_pos[1] - (dims[1]/2-1)).astype(int)
    
    #also load assignments and distinguish between matched and non-matched (for those sessions) - plot in red (matched) vs white (non-matched)
    for n in range(f['A'].shape[1]):
      
          
      if (cm[n,:] > extent[:,0]-10).all() and (cm[n,:] < extent[:,1]+10).all():
        
        c = np.where(assignments[:,s-1]==n)[0]
        if c:
          nMatch = sum(sum(~np.isnan(assignments[c,session[0]-1:session[1]])));
          if nMatch == 3:
            #print(c)
            #print(assignments[c,session[0]-1:session[1]])
            col = 'r'
            lw = 2
          elif nMatch == 2:
            col = 'y'
            lw = 1
          else:
            col = 'r'
            lw = 0.8
          
          ax.contour(norm_nrg(np.reshape(f['A'][:,n].transpose().toarray(),dims,order='F')), levels=[level], colors=col, linewidths=lw)
      
    
    #[pl.contour(norm_nrg(np.reshape(mm.toarray(),dims,order='F')), levels=[level], colors=col, linewidths=1) for mm in f['A'][:,:500].transpose()]
    
    #if s == session[0]:
    text_str = "Session %02d"%s
    y_pos = extent[0,1]+y_shift-7
    #else:
      #text_str = "Session %02d \nShift: (%d,%d)"%(s,x_shift,y_shift)
      #y_pos = extent[0,1]+y_shift-13
    
    
    #ax.plot([210,230],[60,70],'k','LineWidth',5)
    ax.plot([extent[1,0],extent[1,0]]+x_shift,[extent[0,0],extent[0,1]]+y_shift,'k',Linewidth=12)
    ax.plot([extent[1,1],extent[1,1]]+x_shift,[extent[0,0],extent[0,1]]+y_shift,'k',LineWidth=12)
    ax.plot([extent[1,0],extent[1,1]]+x_shift,[extent[0,0],extent[0,0]]+y_shift,'k',LineWidth=12)
    ax.plot([extent[1,0],extent[1,1]]+x_shift,[extent[0,1],extent[0,1]]+y_shift,'k',LineWidth=12)
    
    ax.text(extent[1,0]+x_shift+4,y_pos,text_str,fontsize=16,bbox=dict(facecolor='w', alpha=0.8))
    ax.set_xlim(extent[1,:]+x_shift)
    ax.set_ylim(extent[0,:]+y_shift)
    
    #pl.title('Matches')
    #ax.axis('off')
    #pl.draw()
    pl.show(block=False)
    
    pl.savefig("/media/wollex/Analyze_AS3/Data/879/Figures/ROIs_s=%02d.png"%s)
  #pl.pause(1)
  
  


def norm_nrg(a_):

    a = a_.copy()
    dims = a.shape
    a = a.reshape(-1, order='F')
    indx = np.argsort(a, axis=None)[::-1]
    cumEn = np.cumsum(a.flatten()[indx]**2)
    cumEn /= cumEn[-1]
    a = np.zeros(np.prod(dims))
    a[indx] = cumEn
    return a.reshape(dims, order='F')