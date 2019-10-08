
import h5py, cv2, os
import matplotlib.pyplot as plt

def display_movie_h5(pathSession,fr=15):
    
    for f in os.listdir(pathSession):
      if f.startswith("thy") or f.startswith("shank"):
        fname = pathSession + f
        if f.endswith('.h5'):
          break
    t_wait = int(1/fr*1000)

    with h5py.File(fname, "r") as f:
        for t in range(f['DATA'].shape[0]):
            frame=f['DATA'][t,:,:]
            cv2.imshow('frame', frame)
            #cv2.putText(frame, 'Frame = %d' % t, (5,20), fontFace=5, fontScale=0.8, color=(0, 255, 255), thickness=1)   #(frame.shape[1] // 2 - frame.shape[1] // 10, frame.shape[0]+10)
            cv2.waitKey(t_wait)
            
            #print("%spics/video_t=%04d.png"%(pathSession,t))
            cv2.imwrite("%spics/video_t=%04d.png"%(pathSession,t),frame)
            #pl.savefig()
