"""Lightweight CPU timeline image, synchronized to the selected recorded frame."""
import cv2
import numpy as np


def timeline_image(rows,index,distances):
    image=np.full((300,640,3),245,np.uint8)
    t=np.array([float(r['completed_elapsed_s']) for r in rows]);t-=t[0]
    span=max(float(t[-1]),1.)
    xs=np.rint(55+t/span*565).astype(int)
    def plot(values,color,top,height,low,high):
        values=np.asarray(values,float)
        ys=np.rint(top+height-(np.nan_to_num(values,nan=low)-low)/max(high-low,1e-6)*height).astype(int)
        for i in range(1,len(values)):
            if np.isfinite(values[i-1:i+1]).all():cv2.line(image,(xs[i-1],ys[i-1]),(xs[i],ys[i]),color,2,cv2.LINE_AA)
        if np.isfinite(values[index]):cv2.circle(image,(xs[index],ys[index]),4,color,-1)
    values=np.asarray(distances,float);finite=values[np.isfinite(values)]
    max_distance=max(1.,float(finite.max())*1.1) if len(finite) else 1.
    for y,height,label,upper in ((35,90,'Camera to fixed object (m)',max_distance),(180,85,'C confidence (green) / S final (gold)',1.)):
        cv2.putText(image,label,(55,y-12),cv2.FONT_HERSHEY_SIMPLEX,.43,(40,40,40),1)
        cv2.rectangle(image,(55,y),(620,y+height),(170,170,170),1)
        cv2.putText(image,f'{upper:.2f}',(4,y+5),cv2.FONT_HERSHEY_SIMPLEX,.35,(80,80,80),1)
        cv2.putText(image,'0',(30,y+height),cv2.FONT_HERSHEY_SIMPLEX,.35,(80,80,80),1)
        cv2.line(image,(xs[index],y),(xs[index],y+height),(0,100,240),1)
    plot(values,(220,100,20),35,90,0,max_distance)
    plot([float(r['C']) for r in rows],(50,150,30),180,85,0,1)
    plot([float(r['S']) for r in rows],(0,170,220),180,85,0,1)
    cv2.putText(image,f't={t[index]:.2f}s / {t[-1]:.2f}s',(55,288),cv2.FONT_HERSHEY_SIMPLEX,.4,(30,30,30),1)
    return image[:,:,::-1]
