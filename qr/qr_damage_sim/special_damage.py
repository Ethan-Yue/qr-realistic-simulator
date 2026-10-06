from __future__ import annotations

import cv2
import numpy as np
from .primitives import *
from .params import p, as_pair, irange, frange, color3
from .common_damage import scratch, abrasion, local_fading, glue_stain, droplets, fingerprint, mud, handwriting


def _blank(image):
    return image.copy(), np.zeros(image.shape[:2], np.uint8), {"count": 0}


def tear_or_hole(image, qr_mask, rng, severity="medium", params=None, color=(225,225,225)):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(1,int(2*s))))
    if count<=0:return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.015*s,.055*s]),[.015,.055]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",3)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.82)))
    opacity=float(p(params,"opacity",.98)); out=blend_color(image,mask,color3(p(params,"color_bgr",color)),opacity=opacity,feather_sigma=float(p(params,"feather",.25)))
    edge=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)); out=blend_color(out,edge,color3(p(params,"edge_color_bgr",[85,95,105])),opacity=float(p(params,"edge_opacity",.35)),feather_sigma=.3)
    return out,mask,{"count":count,"radius_ratio":list(rr),"opacity":opacity,"physical_missing":True}


def peel_patch(image, qr_mask, rng, severity="medium", params=None, base_color=(225,225,225)):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(1,int(2*s))))
    if count<=0:return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.012*s,.05*s]),[.012,.05]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",5)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.9)))
    opacity=float(p(params,"opacity",rng.uniform(.75,.98))); out=blend_color(image,mask,color3(p(params,"color_bgr",base_color)),opacity=opacity,feather_sigma=float(p(params,"feather",.5)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"opacity":opacity,"peeling":True}


def rust(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(1,int(2*s))))
    if count<=0:return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.016*s,.065*s]),[.016,.065]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",9)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.7)))
    speck_count=int(p(params,"speckle_count",max(15,int(60*s)))); speck=speckle_mask((h,w),rng,max(1,speck_count),tuple(map(int,as_pair(p(params,"speckle_radius_px",[1,max(2,int(5*s))]),[1,5]))),qr_mask,float(p(params,"speckle_qr_bias",.55))) if speck_count>0 else np.zeros((h,w),np.uint8)
    mask=cv2.bitwise_or(mask,cv2.bitwise_and(speck,cv2.dilate(mask,np.ones((17,17),np.uint8))))
    opacity=float(p(params,"opacity",rng.uniform(.38,.72))); out=blend_color(image,mask,color3(p(params,"color_bgr",[25,70,135])),opacity=opacity,feather_sigma=float(p(params,"feather",1.2)))
    return out,mask,{"count":count,"speckle_count":speck_count,"radius_ratio":list(rr),"opacity":opacity,"corrosion":True}


def oxidation(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(1,int(2*s))))
    if count<=0:return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.025*s,.09*s]),[.025,.09]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1])))
    mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",19)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.65)))
    col=color3(p(params,"color_bgr",[160,165,165] if rng.random()<.5 else [75,85,90])); opacity=float(p(params,"opacity",rng.uniform(.18,.38))); out=blend_color(image,mask,col,opacity=opacity,feather_sigma=float(p(params,"feather",4)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"opacity":opacity,"oxidation":True}


def pitting(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(10,int(45*s))))
    if count<=0:return _blank(image)
    rr=as_pair(p(params,"radius_px",[1,max(2,int(5*s))]),[1,5]); mask=speckle_mask((h,w),rng,count,(max(1,int(rr[0])),max(1,int(rr[1]))),qr_mask,float(p(params,"qr_bias",.8)))
    opacity=float(p(params,"opacity",rng.uniform(.45,.8))); out=blend_color(image,mask,color3(p(params,"color_bgr",[45,50,55])),opacity=opacity,feather_sigma=float(p(params,"feather",.4)))
    return out,mask,{"count":count,"radius_px":list(rr),"opacity":opacity,"pits":True}


def crack(image, qr_mask, rng, severity="medium", params=None, dark=True):
    s=severity_scale(severity); h,w=image.shape[:2]; n=int(p(params,"count",max(1,int(rng.integers(1,4)*s)))); mask=np.zeros((h,w),np.uint8)
    if n<=0:return _blank(image)
    seg_range=as_pair(p(params,"segments",[4,8]),[4,8]); dx=as_pair(p(params,"dx_px",[-25,25]),[-25,25]); dy=as_pair(p(params,"dy_px",[10,35]),[10,35]); width=int(p(params,"width_px",max(1,int(1.5*s))))
    for _ in range(n):
        cx,cy=choose_center(h,w,rng,qr_mask,float(p(params,"qr_bias",.75))); pts=[[cx,cy]]
        for _k in range(max(1,irange(rng,seg_range,[4,8]))):
            px,py=pts[-1]; pts.append([int(np.clip(px+irange(rng,dx,[-25,25]),0,w-1)),int(np.clip(py+irange(rng,dy,[10,35]),0,h-1))])
        cv2.polylines(mask,[np.asarray(pts,np.int32)],False,255,width,cv2.LINE_AA)
    opacity=float(p(params,"opacity",.78)); col=color3(p(params,"color_bgr",[25,30,35] if dark else [235,235,235])); out=blend_color(image,mask,col,opacity=opacity,feather_sigma=float(p(params,"feather",.25)))
    return out,mask,{"count":n,"segments":list(seg_range),"width_px":width,"opacity":opacity}


def fiber_fray(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(8,int(25*s))))
    if count<=0:return _blank(image)
    wr=as_pair(p(params,"width_px",[1,max(1,int(2*s))]),[1,2]); lr=as_pair(p(params,"length_ratio",[.01,.08]),[.01,.08]); mask=scratch_mask((h,w),rng,count,(max(1,int(wr[0])),max(1,int(wr[1]))),lr,qr_mask,float(p(params,"qr_bias",.82)))
    opacity=float(p(params,"opacity",rng.uniform(.35,.7))); out=blend_color(image,mask,color3(p(params,"color_bgr",[205,205,200])),opacity=opacity,feather_sigma=float(p(params,"feather",.3)))
    return out,mask,{"count":count,"width_px":list(wr),"length_ratio":list(lr),"opacity":opacity,"fiber_like":True}


def coating_haze(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",1))
    if count<=0:return _blank(image)
    rr=as_pair(p(params,"radius_ratio",[.04*s,.14*s]),[.04,.14]); r=(max(2,int(min(h,w)*rr[0])),max(3,int(min(h,w)*rr[1]))
    ); mask=irregular_blob_mask((h,w),rng,radius_range=r,n_blobs=(count,count),smooth=int(p(params,"smooth",25)),qr_mask=qr_mask,qr_bias=float(p(params,"qr_bias",.7)))
    opacity=float(p(params,"opacity",rng.uniform(.10,.28))); out=blend_color(image,mask,color3(p(params,"color_bgr",[225,225,225])),opacity=opacity,feather_sigma=float(p(params,"feather",6)))
    return out,mask,{"count":count,"radius_ratio":list(rr),"opacity":opacity,"haze":True}


def dead_pixels(image, qr_mask, rng, severity="medium", params=None, line=False):
    s=severity_scale(severity); h,w=image.shape[:2]; mask=np.zeros((h,w),np.uint8); line=bool(p(params,"line",line))
    if line:
        n=int(p(params,"count",max(1,int(rng.integers(1,4)*s)))); tr=as_pair(p(params,"thickness_px",[1,max(2,int(4*s))]),[1,4])
        if n<=0:return _blank(image)
        for _ in range(n):
            th=max(1,irange(rng,tr,[1,4])); vertical=bool(p(params,"vertical",rng.random()<.5))
            if not vertical: y=int(rng.integers(0,h)); cv2.rectangle(mask,(0,y),(w-1,min(h-1,y+th)),255,-1)
            else: x=int(rng.integers(0,w)); cv2.rectangle(mask,(x,0),(min(w-1,x+th),h-1),255,-1)
    else:
        x0,y0,x1,y1=qr_bbox(qr_mask,h,w); n=int(p(params,"count",max(8,int(rng.integers(20,70)*s)))); sr=as_pair(p(params,"size_px",[1,max(2,int(4*s))]),[1,4])
        if n<=0:return _blank(image)
        for _ in range(n):
            x=int(rng.integers(x0,x1)); y=int(rng.integers(y0,y1)); sz=max(1,irange(rng,sr,[1,4])); cv2.rectangle(mask,(x,y),(min(w-1,x+sz),min(h-1,y+sz)),255,-1)
    opacity=float(p(params,"opacity",.98)); out=blend_color(image,mask,color3(p(params,"color_bgr",[0,0,0])),opacity=opacity,feather_sigma=float(p(params,"feather",0)))
    return out,mask,{"count":n,"line":line,"opacity":opacity}


def toner_wear(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return abrasion(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); count=int(p(params,"count",max(4,int(8*s))))
    if count<=0:return _blank(image)
    pr=as_pair(p(params,"patch_ratio",[.01,.045*s]),[.01,.045]); mask=module_patch_mask(qr_mask,rng,count,pr); target=int(p(params,"target_gray",220)); opacity=float(p(params,"opacity",rng.uniform(.45,.8))); out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",.45)))
    return out,mask,{"count":count,"patch_ratio":list(pr),"target_gray":target,"opacity":opacity,"toner_removed":True}


def inkjet_water_bleed(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return local_fading(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); count=int(p(params,"count",max(3,int(7*s))))
    if count<=0:return _blank(image)
    pr=as_pair(p(params,"patch_ratio",[.012,.055*s]),[.012,.055]); mask=module_patch_mask(qr_mask,rng,count,pr); sigma=float(p(params,"blur_sigma",max(.8,1.8*s))); k=max(3,int(p(params,"kernel_size",5*s))|1); blurred=cv2.GaussianBlur(image,(k,k),sigmaX=sigma); opacity=float(p(params,"opacity",rng.uniform(.45,.78))); a=feather(mask,float(p(params,"feather",max(1.0,2.0*s))))[...,None]*opacity; out=np.clip(image.astype(np.float32)*(1-a)+blurred.astype(np.float32)*a,0,255).astype(np.uint8)
    return out,mask,{"count":count,"patch_ratio":list(pr),"blur_sigma":sigma,"opacity":opacity}


def thermal_band_fade(image, qr_mask, rng, severity="medium", params=None):
    s=severity_scale(severity); h,w=image.shape[:2]; x0,y0,x1,y1=qr_bbox(qr_mask,h,w); n=int(p(params,"count",max(1,int(rng.integers(1,4)*s)))); mask=np.zeros((h,w),np.uint8)
    if n<=0:return _blank(image)
    hr=as_pair(p(params,"band_height_ratio",[.03*s,.12*s]),[.03,.12])
    for _ in range(n):
        bh=max(2,int((y1-y0)*rng.uniform(*hr))); y=int(rng.integers(y0,max(y0+1,y1-bh))); cv2.rectangle(mask,(x0,y),(x1-1,min(h-1,y+bh)),255,-1)
    if qr_mask is not None:mask=cv2.bitwise_and(mask,qr_mask)
    target=int(p(params,"target_gray",rng.integers(185,230))); opacity=float(p(params,"opacity",rng.uniform(.45,.8))); out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",1.2)))
    return out,mask,{"count":n,"band_height_ratio":list(hr),"target_gray":target,"opacity":opacity}


def dot_dropout(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return local_fading(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); h,w=image.shape[:2]; ys,xs=np.where(qr_mask>0); mask=np.zeros((h,w),np.uint8); default=max(10,int(len(xs)*rng.uniform(.001,.004)*s)) if len(xs) else 0; count=int(p(params,"count",default)); rr=as_pair(p(params,"radius_px",[1,max(2,int(3*s))]),[1,3])
    if count<=0:return _blank(image)
    if len(xs):
        ids=rng.choice(len(xs),size=min(count,len(xs)),replace=False)
        for x,y in zip(xs[ids],ys[ids]):cv2.circle(mask,(int(x),int(y)),max(1,irange(rng,rr,[1,3])),255,-1)
        mask=cv2.bitwise_and(mask,qr_mask)
    target=int(p(params,"target_gray",225)); opacity=float(p(params,"opacity",rng.uniform(.55,.9))); out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",.35)))
    return out,mask,{"count":count,"radius_px":list(rr),"target_gray":target,"opacity":opacity,"dot_dropout":True}


def uv_ink_chip(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return peel_patch(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); count=int(p(params,"count",max(3,int(7*s))))
    if count<=0:return _blank(image)
    pr=as_pair(p(params,"patch_ratio",[.008,.035*s]),[.008,.035]); mask=module_patch_mask(qr_mask,rng,count,pr); k=int(p(params,"open_kernel_px",3)); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((max(1,k),max(1,k)),np.uint8)); target=int(p(params,"target_gray",225)); opacity=float(p(params,"opacity",rng.uniform(.55,.9))); out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",.25)))
    return out,mask,{"count":count,"patch_ratio":list(pr),"target_gray":target,"opacity":opacity,"ink_chip":True}


def laser_contrast_loss(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return local_fading(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); count=int(p(params,"count",max(2,int(5*s))))
    if count<=0:return _blank(image)
    pr=as_pair(p(params,"patch_ratio",[.02,.07*s]),[.02,.07]); mask=module_patch_mask(qr_mask,rng,count,pr); sigma=float(p(params,"blur_sigma",max(2,8*s))); local=cv2.GaussianBlur(image,(0,0),sigmaX=sigma); opacity=float(p(params,"opacity",rng.uniform(.45,.75))); a=feather(mask,float(p(params,"feather",1.0)))[...,None]*opacity; out=np.clip(image.astype(np.float32)*(1-a)+local.astype(np.float32)*a,0,255).astype(np.uint8)
    return out,mask,{"count":count,"patch_ratio":list(pr),"blur_sigma":sigma,"opacity":opacity,"contrast_loss":True}


def groove_fill(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return mud(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); count=int(p(params,"count",max(3,int(8*s))))
    if count<=0:return _blank(image)
    pr=as_pair(p(params,"patch_ratio",[.01,.05*s]),[.01,.05]); mask=module_patch_mask(qr_mask,rng,count,pr); opacity=float(p(params,"opacity",rng.uniform(.45,.85))); out=blend_color(image,mask,color3(p(params,"color_bgr",[75,85,90])),opacity=opacity,feather_sigma=float(p(params,"feather",.6)))
    return out,mask,{"count":count,"patch_ratio":list(pr),"opacity":opacity,"groove_contamination":True}


def ink_crack(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return crack(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(3,int(9*s))))
    if count<=0:return _blank(image)
    wr=as_pair(p(params,"width_px",[1,max(1,int(2*s))]),[1,2]); lr=as_pair(p(params,"length_ratio",[.02,.12]),[.02,.12]); lines=scratch_mask((h,w),rng,count,(max(1,int(wr[0])),max(1,int(wr[1]))),lr,qr_mask,float(p(params,"qr_bias",.95))); mask=cv2.bitwise_and(lines,qr_mask); target=int(p(params,"target_gray",225)); opacity=float(p(params,"opacity",rng.uniform(.5,.85))); out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",.25)))
    return out,mask,{"count":count,"width_px":list(wr),"length_ratio":list(lr),"target_gray":target,"opacity":opacity,"ink_crack":True}


def thread_dropout(image, qr_mask, rng, severity="medium", params=None):
    if qr_mask is None:return fiber_fray(image,qr_mask,rng,severity,params)
    s=severity_scale(severity); h,w=image.shape[:2]; count=int(p(params,"count",max(3,int(10*s))))
    if count<=0:return _blank(image)
    wr=as_pair(p(params,"width_px",[1,max(2,int(3*s))]),[1,3]); lr=as_pair(p(params,"length_ratio",[.03,.16]),[.03,.16]); mask=scratch_mask((h,w),rng,count,(max(1,int(wr[0])),max(1,int(wr[1]))),lr,qr_mask,float(p(params,"qr_bias",.95))); mask=cv2.bitwise_and(mask,qr_mask); target=int(p(params,"target_gray",205)); opacity=float(p(params,"opacity",rng.uniform(.55,.88))); out=blend_toward(image,mask,target,opacity=opacity,feather_sigma=float(p(params,"feather",.4)))
    return out,mask,{"count":count,"width_px":list(wr),"length_ratio":list(lr),"target_gray":target,"opacity":opacity,"thread_dropout":True}


def method_fade(image, qr_mask, rng, severity="medium", params=None): return local_fading(image,qr_mask,rng,severity,params)
def method_scuff(image, qr_mask, rng, severity="medium", params=None): return abrasion(image,qr_mask,rng,severity,params)
def screen_line_dropout(image, qr_mask, rng, severity="medium", params=None): return dead_pixels(image,qr_mask,rng,severity,params,line=True)

CARRIER_REGISTRY={
"paper_tear":tear_or_hole,"paper_yellow_spot":local_fading,"paper_fiber_loss":abrasion,"label_edge_peel":peel_patch,"label_missing_piece":tear_or_hole,"label_adhesive_residue":glue_stain,"corrugated_fiber_scuff":abrasion,"corrugated_puncture":tear_or_hole,"corrugated_soil_embed":mud,"film_surface_scuff":scratch,"film_ink_peel":peel_patch,"film_condensation":droplets,"carton_edge_scuff":abrasion,"carton_coating_chip":peel_patch,"metal_rust":rust,"metal_oxidation":oxidation,"metal_pitting":pitting,"plastic_scuff":scratch,"plastic_grease":fingerprint,"plastic_coating_chip":peel_patch,"glass_scuff":coating_haze,"glass_droplets":droplets,"glass_glue_residue":glue_stain,"wood_surface_wear":abrasion,"wood_char_loss":local_fading,"wood_stain":mud,"fabric_fray":fiber_fray,"fabric_thread_loss":fiber_fray,"fabric_hole":tear_or_hole,"ceramic_glaze_chip":peel_patch,"ceramic_crack":crack,"ceramic_stain":mud,"acrylic_haze":coating_haze,"acrylic_scratch":scratch,"acrylic_glue":glue_stain,"poster_tear":tear_or_hole,"poster_uv_fade":local_fading,"poster_graffiti":handwriting,"ticket_crease_wear":abrasion,"ticket_fade":local_fading,"ticket_edge_wear":abrasion,"screen_dead_pixels":dead_pixels,"screen_line_dropout":screen_line_dropout,"screen_smudges":fingerprint}

METHOD_REGISTRY={
"toner_wear":toner_wear,"inkjet_water_bleed":inkjet_water_bleed,"offset_ink_scuff":method_scuff,"thermal_band_fade":thermal_band_fade,"thermal_transfer_flake":uv_ink_chip,"digital_inkjet_fade":method_fade,"flexo_ink_dropout":dot_dropout,"spray_dot_dropout":dot_dropout,"gravure_surface_wear":method_scuff,"uv_ink_chip":uv_ink_chip,"laser_mark_contrast_loss":laser_contrast_loss,"laser_engrave_groove_fill":groove_fill,"screen_ink_crack":ink_crack,"pad_print_wear":toner_wear,"laser_etch_surface_fill":groove_fill,"laser_burn_char_wear":method_fade,"digital_print_fade":method_fade,"jacquard_thread_dropout":thread_dropout,"embroidery_stitch_break":thread_dropout,"underglaze_chip":uv_ink_chip,"photo_surface_scratch":method_scuff,"uv_roll_ink_chip":uv_ink_chip,"display_pixel_dropout":dead_pixels}
