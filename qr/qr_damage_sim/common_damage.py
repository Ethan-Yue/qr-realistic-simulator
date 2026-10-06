from __future__ import annotations

import cv2
import numpy as np
from .primitives import *
from .params import p, as_pair, irange, frange, color3


def _blank(image):
    return image.copy(), np.zeros(image.shape[:2], np.uint8), {"count": 0}


def dust(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]
    auto=int((h*w/1_000_000)*rng.integers(90,220)*s)+12
    count=int(p(params,"count",auto))
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_px",[1,max(2,int(4*s))]),[1,4])
    mask=speckle_mask((h,w),rng,count,(max(1,int(rr[0])),max(1,int(rr[1]))),qr_mask,float(p(params,"qr_bias",.4)))
    gray=int(p(params,"gray",rng.integers(95,165)))
    opacity=float(p(params,"opacity",rng.uniform(.22,.50)))
    out=blend_color(image,mask,(gray,)*3,opacity=opacity,feather_sigma=float(p(params,"feather",.5)))
    return out,mask,{"count":count,"radius_px":list(rr),"opacity":opacity,"qr_bias":float(p(params,"qr_bias",.4))}


def mud(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]
    count=int(p(params,"count",max(1,int(2*s))))
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.015*s,.06*s]),[.015,.06]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",9)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.7)))
    opacity=float(p(params,"opacity",rng.uniform(.45,.78))); col=color3(p(params,"color_bgr",[40,65,85]))
    out=blend_color(image,mask,col,opacity=opacity,feather_sigma=float(p(params,"feather",1.5)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"opacity":opacity}


def oil_stain(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(1,int(2*s))))
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.03*s,.11*s]),[.03,.11]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",15)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.65)))
    opacity=float(p(params,"opacity",rng.uniform(.16,.34)*s)); out=blend_color(image,mask,color3(p(params,"color_bgr",[25,45,55])),opacity=opacity,feather_sigma=float(p(params,"feather",3.0)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"opacity":opacity}


def water_stain(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1))
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.025*s,.09*s]),[.025,.09]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    filled=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",15)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.7)))
    ew=max(3,int(p(params,"ring_width_px",5*s)))|1; inner=cv2.erode(filled,np.ones((ew,ew),np.uint8),iterations=1); ring=cv2.subtract(filled,inner)
    out=blend_color(image,filled,color3(p(params,"fill_color_bgr",[150,170,175])),opacity=float(p(params,"fill_opacity",rng.uniform(.06,.16)*s)),feather_sigma=float(p(params,"fill_feather",5)))
    out=blend_color(out,ring,color3(p(params,"ring_color_bgr",[95,115,125])),opacity=float(p(params,"ring_opacity",rng.uniform(.12,.25)*s)),feather_sigma=float(p(params,"ring_feather",1.5)))
    return out,(filled>0).astype(np.uint8)*255,{"count":count,"radius_ratio":list(rr),"ring_width_px":ew}


def fingerprint(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1))
    if count<=0: return _blank(image)
    mask=np.zeros((h,w),np.uint8); ridge_count=int(p(params,"ridge_count",max(7,int(14*s)))); rr=as_pair(p(params,"radius_ratio",[.05*s,.10*s]),[.05,.10]); rs=max(1.0,min(h,w)/1024.0)
    for _ in range(count):
        cx,cy=choose_center(h,w,rng,qr_mask,float(p(params,"qr_bias",.72))); base=int(min(h,w)*rng.uniform(*rr))
        angle=float(p(params,"angle_deg",rng.uniform(-25,25)))
        for k in range(ridge_count):
            axes=(max(4,base+k*max(1,int(base*.07))),max(3,int((base+k*base*.07)*.58)))
            cv2.ellipse(mask,(cx,cy),axes,angle,195,345,255,max(1,int(float(p(params,"line_width_px",1.2*s*rs)))),cv2.LINE_AA)
    opacity=float(p(params,"opacity",rng.uniform(.08,.20))); out=blend_color(image,mask,color3(p(params,"color_bgr",[70,80,85])),opacity=opacity,feather_sigma=float(p(params,"feather",.7)))
    return out,mask,{"count":count,"ridge_count":ridge_count,"radius_ratio":list(rr),"opacity":opacity}


def scratch(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(1,int(rng.integers(2,7)*s))))
    if count<=0: return _blank(image)
    wr=as_pair(p(params,"width_px",[1,max(2,int(3*s))]),[1,3]); lr=as_pair(p(params,"length_ratio",[.07,.32]),[.07,.32])
    mask=scratch_mask((h,w),rng,count,(max(1,int(wr[0])),max(1,int(wr[1]))),lr,qr_mask,float(p(params,"qr_bias",.8)))
    light=bool(p(params,"light",rng.random()<.65)); col=color3(p(params,"color_bgr",[225,225,225] if light else [35,35,35])); opacity=float(p(params,"opacity",rng.uniform(.35,.8)))
    out=blend_color(image,mask,col,opacity=opacity,feather_sigma=float(p(params,"feather",.35)))
    return out,mask,{"count":count,"width_px":list(wr),"length_ratio":list(lr),"opacity":opacity,"light":light}


def abrasion(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(1,int(2*s))))
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.018*s,.07*s]),[.018,.07]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    blob=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",7)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.78)))
    sc=int(p(params,"micro_scratch_count",max(4,int(12*s)))); scratches=scratch_mask((h,w),rng,max(1,sc),(1,max(2,int(2*s))),(0.02,.13),qr_mask,.8) if sc>0 else np.zeros((h,w),np.uint8)
    mask=cv2.bitwise_and(cv2.bitwise_or(blob,scratches),cv2.dilate(blob,np.ones((5,5),np.uint8),iterations=1)); target=int(p(params,"target_gray",185)); opacity=float(p(params,"opacity",rng.uniform(.22,.48)*s))
    out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",.8)))
    return out,mask,{"count":count,"micro_scratch_count":sc,"radius_ratio":list(rr),"target_gray":target,"opacity":opacity}


def local_fading(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1))
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.04*s,.13*s]),[.04,.13]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",21)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.82)))
    target=int(p(params,"target_gray",rng.integers(195,238))); opacity=float(p(params,"opacity",rng.uniform(.20,.48)*s)); out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",5)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"target_gray":target,"opacity":opacity}


def tape(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1)); out=image.copy(); mask=np.zeros((h,w),np.uint8)
    if count<=0: return _blank(image)
    sr=as_pair(p(params,"size_ratio",[.10*s,.25*s]),[.10,.25]); opacity=float(p(params,"opacity",rng.uniform(.16,.35))); col=color3(p(params,"color_bgr",[185,205,205] if rng.random()<.75 else [110,150,180]))
    for _ in range(count):
        m=random_rect_mask((h,w),rng,sr,qr_mask,float(p(params,"qr_bias",.78)),bool(p(params,"angle",True))); mask=cv2.bitwise_or(mask,m); out=blend_color(out,m,col,opacity=opacity,feather_sigma=float(p(params,"feather",.7)))
        edge=cv2.morphologyEx(m,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)); out=blend_color(out,edge,(95,105,110),opacity=float(p(params,"edge_opacity",.22)),feather_sigma=.4)
    return out,mask,{"count":count,"size_ratio":list(sr),"opacity":opacity}


def sticker(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1)); out=image.copy(); mask=np.zeros((h,w),np.uint8)
    if count<=0: return _blank(image)
    sr=as_pair(p(params,"size_ratio",[.07*s,.18*s]),[.07,.18]); palette=[
        [239,239,239],  # white paper label
        [215,229,240],  # warm ivory
        [227,231,231],  # pale grey label stock
        [218,229,225],  # lightly tinted matte stock
    ]
    opacity=float(p(params,"opacity",rng.uniform(.90,1.0)))
    edge_opacity=float(p(params,"edge_opacity",.22))
    for _ in range(count):
        m=random_rect_mask((h,w),rng,sr,qr_mask,float(p(params,"qr_bias",.84)),bool(p(params,"angle",True)))
        mask=cv2.bitwise_or(mask,m)
        col=color3(p(params,"color_bgr",palette[int(rng.integers(0,len(palette)))]))
        out=blend_color(out,m,col,opacity=opacity,feather_sigma=float(p(params,"feather",.25)))
        # A thin in-mask rim gives the otherwise flat occluder a paper edge.
        inner=cv2.erode(m,np.ones((3,3),np.uint8),iterations=1)
        rim=cv2.subtract(m,inner)
        rim_col=tuple(max(0,int(v)-32) for v in col)
        out=blend_color(out,rim,rim_col,opacity=edge_opacity,feather_sigma=0.0)
    return out,mask,{"count":count,"size_ratio":list(sr),"opacity":opacity,"edge_opacity":edge_opacity}


def handwriting(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; rs=max(1.0,min(h,w)/1024.0); mask=np.zeros((h,w),np.uint8); n=int(p(params,"count",max(1,int(rng.integers(2,5)*s))))
    if n<=0: return _blank(image)
    x0,y0,x1,y1=qr_bbox(qr_mask,h,w); point_range=as_pair(p(params,"points_per_stroke",[3,7]),[3,7]); width=as_pair(p(params,"width_px",[1,max(1,int(4*s*rs))]),[1,4])
    for _ in range(n):
        if qr_mask is not None and rng.random()<float(p(params,"qr_bias",.8)): sx=int(rng.integers(x0,x1)); sy=int(rng.integers(y0,y1))
        else: sx,sy=choose_center(h,w,rng,None,0)
        pts=[]; npnt=max(2,irange(rng,point_range,[3,7]))
        for k in range(npnt): pts.append([int(np.clip(sx+k*rng.integers(8,24),0,w-1)),int(np.clip(sy+rng.normal(0,float(p(params,"vertical_jitter_px",12*s))),0,h-1))])
        cv2.polylines(mask,[np.array(pts,np.int32)],False,255,max(1,irange(rng,width,[1,4])),cv2.LINE_AA)
    col=color3(p(params,"color_bgr",[10,20,25] if rng.random()<.65 else [80,35,150])); opacity=float(p(params,"opacity",rng.uniform(.75,1.0))); out=blend_color(image,mask,col,opacity=opacity,feather_sigma=float(p(params,"feather",.2)))
    return out,mask,{"count":n,"width_px":list(width),"opacity":opacity}


def stamp(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1)); mask=np.zeros((h,w),np.uint8)
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.035*s,.075*s]),[.035,.075]); width=int(p(params,"line_width_px",max(1,int(2*s))))
    radii=[]
    for _ in range(count):
        cx,cy=choose_center(h,w,rng,qr_mask,float(p(params,"qr_bias",.8))); r=max(3,int(min(h,w)*rng.uniform(*rr))); radii.append(r); cv2.circle(mask,(cx,cy),r,255,width,cv2.LINE_AA); cv2.line(mask,(cx-r,cy),(cx+r,cy),255,width,cv2.LINE_AA)
    opacity=float(p(params,"opacity",rng.uniform(.45,.75))); out=blend_color(image,mask,color3(p(params,"color_bgr",[30,35,180])),opacity=opacity,feather_sigma=float(p(params,"feather",.4)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"radii_px":radii,"opacity":opacity}


def glue_stain(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1))
    if count<=0: return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.018*s,.07*s]),[.018,.07]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",17)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.65)))
    opacity=float(p(params,"opacity",rng.uniform(.08,.22))); out=blend_color(image,mask,color3(p(params,"color_bgr",[140,175,195])),opacity=opacity,feather_sigma=float(p(params,"feather",3)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"opacity":opacity}


def droplets(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; n=int(p(params,"count",max(3,int(rng.integers(6,20)*s)))); mask=np.zeros((h,w),np.uint8); out=image.copy()
    if n<=0: return _blank(image)
    rr=as_pair(p(params,"radius_px",[3,max(4,int(min(h,w)*.018*s))]),[3,10])
    for _ in range(n):
        cx,cy=choose_center(h,w,rng,qr_mask,float(p(params,"qr_bias",.5))); r=max(1,irange(rng,rr,[3,10])); cv2.circle(mask,(cx,cy),r,255,-1,cv2.LINE_AA); cv2.circle(out,(cx,cy),r,color3(p(params,"edge_color_bgr",[215,225,230])),max(1,r//5),cv2.LINE_AA); cv2.circle(out,(max(0,cx-r//3),max(0,cy-r//3)),max(1,r//5),color3(p(params,"highlight_bgr",[245,245,245])),-1,cv2.LINE_AA)
    out=blend_color(out,mask,color3(p(params,"fill_color_bgr",[205,215,220])),opacity=float(p(params,"opacity",.07)),feather_sigma=float(p(params,"feather",.7)))
    return out,mask,{"count":n,"radius_px":list(rr),"opacity":float(p(params,"opacity",.07))}


COMMON_REGISTRY={"dust":dust,"mud":mud,"oil_stain":oil_stain,"water_stain":water_stain,"fingerprint":fingerprint,"scratch":scratch,"abrasion":abrasion,"local_fading":local_fading,"tape":tape,"sticker":sticker,"handwriting":handwriting,"stamp":stamp,"glue_stain":glue_stain,"droplets":droplets}
