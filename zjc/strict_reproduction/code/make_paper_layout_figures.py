#!/usr/bin/env python3
"""Rebuild the non-error-statistics figures of the Tianjin paper as Chinese SVGs.

The page geometry follows the source paper.  Dense SAR layers are embedded as
high-resolution rasters, while labels, diagrams, axes, arrows and legends stay
as editable SVG vectors.  No missing height is filled from ``Floor``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


W, H = 10_000, 7_000
PAPER_BBOX = (117.15, 39.10, 117.24, 39.16)
RADAR_CROP = (4300, 7300, 3300, 5300)
CM = 1 / 2.54

mpl.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.sans-serif": ["Noto Sans CJK SC"],
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "mathtext.default": "regular",
    "image.composite_image": False,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "font.size": 7.5,
})


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.035,
                facecolor="white", dpi=300)
    plt.close(fig)


def amp_image(path: Path, stride: int = 2) -> np.ndarray:
    a = np.memmap(path, dtype=">f4", mode="r", shape=(H, W))[::stride, ::stride]
    z = np.log1p(np.asarray(a, dtype=np.float32))
    ok = np.isfinite(z) & (z > 0)
    lo, hi = np.quantile(z[ok], (0.01, 0.997))
    z = np.clip((z - lo) / (hi - lo), 0, 1)
    return np.power(z, 0.70, dtype=np.float32)


def sar(ax, amp, crop=None, stride=2, labels=True):
    if crop is None:
        ext = (0, W, H, 0)
        ax.imshow(amp, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest", rasterized=True)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
    else:
        x0, x1, y0, y1 = crop
        sub = amp[y0 // stride:y1 // stride, x0 // stride:x1 // stride]
        ax.imshow(sub, cmap="gray", vmin=0, vmax=1, extent=(x0, x1, y1, y0), interpolation="nearest", rasterized=True)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    if labels:
        ax.set_xlabel("距离向像元"); ax.set_ylabel("方位向像元")
    else:
        ax.set_axis_off()


def panel(ax, text):
    """Put panel text inside the axes so it never collides with axis labels."""
    ax.text(
        0.018, 0.975, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=7.4, fontweight="bold", zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
    )


def box(ax, xy, wh, text, fc="#f7f7f7", ec="#222", fs=8, dashed=False):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015", facecolor=fc,
                       edgecolor=ec, linewidth=1.0, linestyle="--" if dashed else "-")
    ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)
    return p


def arrow(ax, p0, p1, color="#222", lw=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, color=color))


def fig31(out):
    fig, ax = plt.subplots(figsize=(16*CM, 22*CM)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    box(ax,(.08,.90),(.34,.065),"时序 SAR 影像",fc="#dce6f1"); box(ax,(.58,.90),(.34,.065),"二维建筑轮廓矢量",fc="#dce6f1")
    box(ax,(.08,.79),(.34,.065),"影像配准与差分干涉处理"); box(ax,(.58,.79),(.34,.065),"建筑属性筛选与坐标转换")
    arrow(ax,(.25,.90),(.25,.855)); arrow(ax,(.75,.90),(.75,.855))
    box(ax,(.06,.46),(.88,.28),"",fc="white",ec="#555",dashed=True)
    ax.text(.085,.715,"建筑物区域独立相位解缠",fontsize=9,weight="bold")
    steps=[("建筑轮廓投影至雷达坐标",.64),("DBSCAN 提取独立建筑区域",.57),("振幅离差与相干性筛选",.50)]
    for t,y in steps: box(ax,(.18,y),(.64,.045),t,fc="#f6f6f6");
    arrow(ax,(.25,.79),(.37,.685)); arrow(ax,(.75,.79),(.63,.685)); arrow(ax,(.50,.64),(.50,.615)); arrow(ax,(.50,.57),(.50,.545)); arrow(ax,(.50,.50),(.50,.43))
    box(ax,(.18,.37),(.64,.055),"逐建筑 GAMMA-MCF 相位解缠",fc="#fff2cc")
    arrow(ax,(.50,.37),(.50,.335)); box(ax,(.18,.275),(.64,.055),"SBAS 高程残差反演",fc="#e2f0d9")
    arrow(ax,(.50,.275),(.50,.24)); box(ax,(.18,.18),(.64,.055),"建筑级稳健聚合与质量控制",fc="#e2f0d9")
    arrow(ax,(.50,.18),(.50,.145)); box(ax,(.18,.085),(.64,.055),"城市建筑三维重建",fc="#d9eaf7")
    save(fig,out)


def fig32(out):
    fig, ax = plt.subplots(figsize=(16*CM,8*CM)); ax.set_xlim(0,16); ax.set_ylim(0,7); ax.axis("off")
    polys=[[(.8,1),(3.2,1.1),(3.7,2.8),(1.2,3.1)],[(1.5,4),(3.6,3.7),(4.1,5.3),(2.1,6)],[(4.2,1.2),(6,1.8),(5.5,3.4),(4,2.9)]]
    for i,p in enumerate(polys): ax.add_patch(Polygon(p,facecolor=["#f4a261","#2a9d8f","#457b9d"][i],alpha=.75,edgecolor="#333"))
    ax.text(3.1,.25,"地理坐标系建筑轮廓",ha="center")
    arrow(ax,(6.3,3.5),(9.4,3.5),lw=1.5); ax.text(7.85,3.85,"二维查找表",ha="center"); ax.text(7.85,3.1,"地理坐标 → 雷达坐标",ha="center",fontsize=7)
    for i,p in enumerate(polys):
        q=[(9.4+(x-.5)*.8, .5+(y-.2)*.85) for x,y in p]
        ax.add_patch(Polygon(q,facecolor=["#f4a261","#2a9d8f","#457b9d"][i],alpha=.75,edgecolor="#333"))
    ax.annotate("距离向",(15.1,.55),(12.6,.55),arrowprops={"arrowstyle":"->"},ha="center")
    ax.annotate("方位向",(12.1,6.4),(12.1,4.6),arrowprops={"arrowstyle":"->"},ha="center",rotation=90)
    ax.text(12.3,.25,"雷达距离—方位坐标系",ha="center")
    save(fig,out)


def fig33(out):
    rng=np.random.default_rng(12); c1=rng.normal((3.4,3.8),(.55,.45),(34,2)); c2=rng.normal((8.0,2.4),(.70,.55),(40,2)); noise=np.array([[1,1],[5.8,5.5],[10.4,5.1],[10.8,1.0]])
    fig,ax=plt.subplots(figsize=(15*CM,8*CM)); ax.scatter(*c1.T,s=23,c="#3182bd",edgecolor="white",lw=.3); ax.scatter(*c2.T,s=23,c="#31a354",edgecolor="white",lw=.3); ax.scatter(*noise.T,s=35,c="#d62728",marker="x")
    ax.add_patch(Circle(c1[3],1.05,fill=False,ls="--",ec="#555")); ax.annotate("邻域半径 ε",c1[3],(1.3,6),arrowprops={"arrowstyle":"->"})
    ax.scatter(*c1[3],s=65,facecolor="#ffdd55",edgecolor="#111",label="核心点"); ax.scatter(*c2[15],s=55,facecolor="white",edgecolor="#111",label="边界点"); ax.scatter([],[],marker="x",c="#d62728",label="噪声点")
    ax.set_xlim(0,12);ax.set_ylim(0,7);ax.set_aspect("equal");ax.set_xlabel("距离向");ax.set_ylabel("方位向");ax.legend(frameon=False,ncol=3,loc="upper right"); save(fig,out)


def fig34(amp,out):
    fig,ax=plt.subplots(figsize=(15*CM,10*CM)); sar(ax,amp,labels=False)
    roi=np.array([[3100,1900],[7850,1200],[8600,5150],[3750,5850]])
    ax.add_patch(Polygon(roi,fill=False,edgecolor="#ff3030",lw=1.5)); ax.text(5000,1300,"天津中心城区实验区",c="white",path_effects=[pe.withStroke(linewidth=2,foreground="black")],fontsize=9)
    ax.annotate("N",(.93,.92),(.93,.78),xycoords="axes fraction",textcoords="axes fraction",ha="center",fontsize=10,weight="bold",arrowprops={"arrowstyle":"-|>","lw":1.2,"color":"white"},color="white")
    ax.plot([.72,.82,.92],[.06,.06,.06],transform=ax.transAxes,c="white",lw=2); ax.text(.82,.075,"约 5 km",transform=ax.transAxes,ha="center",c="white",path_effects=[pe.withStroke(linewidth=1.5,foreground="black")])
    save(fig,out)


def baseline(path,out):
    rows=[]
    for line in path.read_text().splitlines():
        f=line.split();
        if f: rows.append(f)
    nodes={}
    edges=[]
    for f in rows:
        d1,d2=f[1],f[2]; b1,b2=float(f[7]),float(f[8]); nodes[d1]=b1;nodes[d2]=b2;edges.append((d1,d2))
    dates=sorted(nodes); x={d:i for i,d in enumerate(dates)}
    fig,ax=plt.subplots(figsize=(16*CM,8*CM))
    for a,b in edges: ax.plot([x[a],x[b]],[nodes[a],nodes[b]],c="#222",lw=.55,zorder=1)
    ax.scatter([x[d] for d in dates],[nodes[d] for d in dates],c="#d7191c",s=22,zorder=2)
    ax.set_xticks(range(len(dates)),[d[4:6]+"-"+d[6:] for d in dates],rotation=72,ha="right",fontsize=5.5);ax.set_ylabel("垂直基线（m）");ax.set_xlabel("日期（2023—2024年）",labelpad=7);ax.grid(ls="--",lw=.35,alpha=.55);fig.subplots_adjust(left=.13,right=.98,top=.96,bottom=.25);save(fig,out)


def add_north_scale(ax):
    ax.annotate("N",(.94,.94),(.94,.82),xycoords="axes fraction",textcoords="axes fraction",ha="center",weight="bold",arrowprops={"arrowstyle":"-|>","lw":1})
    ax.plot([.73,.91],[.08,.08],transform=ax.transAxes,c="#111",lw=2); ax.text(.82,.095,"2000 m",transform=ax.transAxes,ha="center",fontsize=6)


def fig36(gdf,floor_rdc,out):
    x0,y0,x1,y1=PAPER_BBOX; sub=gdf.cx[x0:x1,y0:y1].copy()
    bounds=[.5,5.5,10.5,15.5,20.5,30.5]; colors=mpl.colormaps["turbo"](np.linspace(.08,.92,5))
    cmap=mpl.colors.ListedColormap(colors); norm=mpl.colors.BoundaryNorm(bounds,cmap.N)
    fig=plt.figure(figsize=(15*CM,18*CM));gs=fig.add_gridspec(2,2,width_ratios=[1,.045],hspace=.27,wspace=.06)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[1,0]);cax=fig.add_subplot(gs[:,1])
    sub.plot(ax=a,column="Floor",cmap=cmap,norm=norm,edgecolor="none",rasterized=True)
    a.set_xlim(x0,x1);a.set_ylim(y0,y1);a.set_aspect(1/np.cos(np.deg2rad((y0+y1)/2)));a.set_xlabel("经度");a.set_ylabel("纬度");add_north_scale(a);panel(a,"（a）地理坐标系")
    r=np.memmap(floor_rdc,dtype=">f4",mode="r",shape=(H,W))[3300:5700,4300:8100];m=np.ma.masked_where(r<=0,r)
    b.imshow(m,cmap=cmap,norm=norm,extent=(0,3800,2400,0),interpolation="nearest",rasterized=True)
    b.set_aspect("equal");b.set_xlabel("距离向像元");b.set_ylabel("方位向像元");panel(b,"（b）雷达坐标系")
    sm=mpl.cm.ScalarMappable(norm=norm,cmap=cmap);cb=fig.colorbar(sm,cax=cax,ticks=[3,8,13,18,25])
    cb.ax.set_yticklabels(["1—5","6—10","11—15","16—20","21—30"]);cb.set_label("楼层数（仅用于候选建筑几何复现）")
    save(fig,out)


def scatter_mask(ax,d,color,size=.25,alpha=.7,key=None,crop=None):
    m=np.ones(len(d["row"]),bool)
    if key is not None:m &= d[key]
    if crop:
        x0,x1,y0,y1=crop;m &= (d["col"]>=x0)&(d["col"]<x1)&(d["row"]>=y0)&(d["row"]<y1)
    ax.scatter(d["col"][m],d["row"][m],s=size,c=color,alpha=alpha,lw=0,rasterized=True)


def fig37(amp,search,quality,out):
    fig=plt.figure(figsize=(18*CM,10*CM));gs=fig.add_gridspec(1,2,width_ratios=[1,1.12],wspace=.04);a=fig.add_subplot(gs[0]);b=fig.add_subplot(gs[1]);sar(a,amp,labels=False);scatter_mask(a,search,"#ef2b2d",.08,.55);a.add_patch(Rectangle((4300,3300),3000,2000,fill=False,ec="#51c46b",lw=1.4));sar(b,amp,RADAR_CROP,labels=False);scatter_mask(b,search,"#ef2b2d",.35,.65,crop=RADAR_CROP);scatter_mask(b,quality,"#00e5ff",.55,.85,"paper_quality_selected",RADAR_CROP)
    for yy in (.35,.65): fig.add_artist(mpl.lines.Line2D([.46,.515],[yy,yy],transform=fig.transFigure,c="#51c46b",lw=.8))
    b.legend(handles=[mpl.lines.Line2D([],[],marker="o",ls="",c="#ef2b2d",label="独立建筑区域"),mpl.lines.Line2D([],[],marker="o",ls="",c="#00e5ff",label="论文阈值像元")],loc="lower right",fontsize=6,facecolor="white",framealpha=.85);save(fig,out)


def grid_points(d,value,crop):
    x0,x1,y0,y1=crop; arr=np.full((y1-y0,x1-x0),np.nan,np.float32);m=(d["col"]>=x0)&(d["col"]<x1)&(d["row"]>=y0)&(d["row"]<y1);arr[d["row"][m]-y0,d["col"][m]-x0]=value[m];return arr


def robust_phase_norm(*arrays):
    """Return one robust colour scale for genuinely comparable phase panels."""
    values=np.concatenate([z[np.isfinite(z)] for z in arrays if np.any(np.isfinite(z))])
    lo,hi=np.quantile(values,(.015,.985))
    if not np.isfinite(lo+hi) or hi<=lo: lo,hi=float(np.nanmin(values)),float(np.nanmax(values))
    return Normalize(float(lo),float(hi))


def fig38(amp,obs,old,ind,out):
    crop=RADAR_CROP; wrap=grid_points(obs,obs["filtered_wrapped_phase_rad"],crop); u0=grid_points(old,old["unwrapped_phase_far_ground_zero_rad"],crop);u1=grid_points(ind,ind["unwrapped_phase_far_ground_zero_rad"],crop)
    phase_norm=robust_phase_norm(u0,u1)
    fig,axs=plt.subplots(2,2,figsize=(17*CM,11.8*CM),gridspec_kw={"hspace":.10,"wspace":.06})
    sar(axs[0,0],amp,crop,labels=False);panel(axs[0,0],"（a）雷达幅度")
    sar(axs[0,1],amp,crop,labels=False);iw=axs[0,1].imshow(wrap,cmap="twilight",vmin=-np.pi,vmax=np.pi,extent=(crop[0],crop[1],crop[3],crop[2]),interpolation="nearest",alpha=.96,rasterized=True);panel(axs[0,1],"（b）建筑支持域内滤波缠绕相位")
    for ax,z,t in [(axs[1,0],u0,"（c）覆盖式 MCF 对照"),(axs[1,1],u1,"（d）建筑区域独立 MCF")]:
        sar(ax,amp,crop,labels=False);ip=ax.imshow(z,cmap="turbo",norm=phase_norm,extent=(crop[0],crop[1],crop[3],crop[2]),interpolation="nearest",alpha=.96,rasterized=True);panel(ax,t)
    cbw=fig.colorbar(iw,ax=axs[0,1],fraction=.032,pad=.018);cbw.set_label("缠绕相位（rad）")
    cbp=fig.colorbar(ip,ax=axs[1,:],fraction=.020,pad=.018);cbp.set_label("解缠相位（rad）")
    save(fig,out)


def choose_uids(obs):
    uid=obs["building_uid"].astype(int); counts=dict(zip(*np.unique(uid,return_counts=True))); one=max(counts,key=counts.get)
    centers=[]
    for u in counts:
        m=uid==u;centers.append((u,float(np.median(obs["col"][m])),float(np.median(obs["row"][m])),counts[u]))
    best=None
    for a in centers:
        for b in centers:
            if a[0]>=b[0]:continue
            dist=np.hypot(a[1]-b[1],a[2]-b[2]);score=(a[3]+b[3])/(dist+8)
            if dist<150 and (best is None or score>best[0]):best=(score,a[0],b[0])
    return one, (best[1],best[2]) if best else tuple(sorted(counts,key=counts.get,reverse=True)[:2])


def local_case(obs,old,ind,uids,out,title):
    m=np.isin(obs["building_uid"],uids);x=obs["col"][m];y=obs["row"][m];pad=15;crop=(int(x.min()-pad),int(x.max()+pad+1),int(y.min()-pad),int(y.max()+pad+1));w=grid_points(obs,obs["filtered_wrapped_phase_rad"],crop);u0=grid_points(old,old["unwrapped_phase_far_ground_zero_rad"],crop);u1=grid_points(ind,ind["unwrapped_phase_far_ground_zero_rad"],crop)
    phase_norm=robust_phase_norm(u0,u1)
    fig,axs=plt.subplots(2,2,figsize=(15*CM,10.5*CM),gridspec_kw={"hspace":.26,"wspace":.22});extent=(crop[0],crop[1],crop[3],crop[2]);
    for ax,z,t,cmap,norm in [(axs[0,0],w,"（a）缠绕相位","twilight",Normalize(-np.pi,np.pi)),(axs[0,1],u0,"（b）覆盖式 MCF 对照","turbo",phase_norm),(axs[1,0],u1,"（c）建筑区域独立 MCF","turbo",phase_norm)]:ax.imshow(z,cmap=cmap,norm=norm,extent=extent,interpolation="nearest",rasterized=True);ax.set_aspect("equal");ax.set_xlabel("距离向像元");ax.set_ylabel("方位向像元");panel(ax,t)
    mi=np.isin(ind["building_uid"],uids); mo=np.isin(old["building_uid"],uids)
    vals1=ind["unwrapped_phase_far_ground_zero_rad"][mi]; vals0=old["unwrapped_phase_far_ground_zero_rad"][mo]
    ord1=np.argsort(ind["col"][mi]); ord0=np.argsort(old["col"][mo])
    axs[1,1].plot(np.linspace(0,1,len(vals1)),vals1[ord1],c="#d7191c",lw=1.1,label="独立区域 MCF");axs[1,1].plot(np.linspace(0,1,len(vals0)),vals0[ord0],c="#2c7bb6",lw=.9,ls="--",label="覆盖式对照");axs[1,1].set_xlabel("归一化距离向位置");axs[1,1].set_ylabel("解缠相位（rad）");axs[1,1].grid(ls="--",lw=.35,alpha=.6);axs[1,1].legend(fontsize=6,loc="best");panel(axs[1,1],"（d）相位剖面对比");fig.suptitle(title,fontsize=10,y=.995);save(fig,out)


def fig41(out):
    fig,ax=plt.subplots(figsize=(10*CM,18*CM));ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off");ys=[.84,.66,.48,.30,.12];ts=["建筑区域独立解缠相位","干涉图质量筛选与基线网络","单体建筑 SBAS 高程反演","建筑矢量拉伸为三维白模","天津中心城区三维模型"]
    cols=["#d9eaf7","#fff2cc","#e2f0d9","#eadcf8","#f7d9c4"]
    for y,t,c in zip(ys,ts,cols):box(ax,(.12,y),(.76,.10),t,fc=c,fs=9)
    for y in ys[:-1]:arrow(ax,(.50,y),(.50,y-.07))
    save(fig,out)


def fig42(amp,pix_good,pix_bad,out):
    crop=RADAR_CROP;fig,axs=plt.subplots(2,1,figsize=(14*CM,13*CM),gridspec_kw={"hspace":.25});
    for ax,d,t in [(axs[0],pix_good,"（a）建筑区域独立 MCF"),(axs[1],pix_bad,"（b）任意首像元零点对照")]:
        sar(ax,amp,crop,labels=False);v=d["dem_error_or_height_above_anchor_m"];m=(d["col"]>=crop[0])&(d["col"]<crop[1])&(d["row"]>=crop[2])&(d["row"]<crop[3]);sc=ax.scatter(d["col"][m],d["row"][m],c=v[m],s=.8,cmap="jet",norm=Normalize(0,220),lw=0,rasterized=True);panel(ax,t)
    cb=fig.colorbar(sc,ax=axs,fraction=.027,pad=.02);cb.set_label("建筑像元相对高差（m）");save(fig,out)


def geo_height(ax,gdf,col,vmax,title):
    x0,y0,x1,y1=PAPER_BBOX;sub=gdf.cx[x0:x1,y0:y1];base=sub[sub[col].isna()];sol=sub[sub[col].notna()];base.plot(ax=ax,facecolor="#eef3f7",edgecolor="#bdd2e5",lw=.08,rasterized=True);sol.plot(ax=ax,column=col,cmap="Spectral_r",vmin=0,vmax=vmax,edgecolor="none",rasterized=True);ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect(1/np.cos(np.deg2rad(39.13)));ax.set_title(title,fontsize=8);ax.set_xlabel("经度");ax.set_ylabel("纬度");add_north_scale(ax)


def neutral_footprints(ax,gdf,bbox,title,missing=False):
    """Draw geometry only; never imply unavailable external height values."""
    x0,y0,x1,y1=bbox;sub=gdf.cx[x0:x1,y0:y1]
    sub.plot(ax=ax,facecolor="#e9edf1",edgecolor="#9aa9b6",lw=.10,rasterized=True)
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect(1/np.cos(np.deg2rad(39.13)))
    ax.set_title(title,fontsize=8);ax.set_xlabel("经度");ax.set_ylabel("纬度");add_north_scale(ax)
    if missing:
        ax.text(.5,.5,"原始数据未提供\n本图不绘制高度值",transform=ax.transAxes,
                ha="center",va="center",fontsize=8,color="#7d1f1f",
                bbox={"fc":"white","alpha":.94,"ec":"#b65c5c","pad":4})


def fig48(gdf,out):
    fig,axs=plt.subplots(3,2,figsize=(17*CM,22*CM),gridspec_kw={"width_ratios":[1.55,.65],"hspace":.26,"wspace":.08});col="recommended_building_height_m";vmax=150
    detail=(117.195,39.118,117.215,39.137)
    missing_titles=["（a）GBA 参考产品（数据未提供）","（b）CNBH-10 m 参考产品（数据未提供）"]
    for r,title in enumerate(missing_titles):
        neutral_footprints(axs[r,0],gdf,PAPER_BBOX,title,missing=True)
        axs[r,0].set_xlabel("")
        neutral_footprints(axs[r,1],gdf,detail,"局部放大",missing=False);axs[r,1].set_axis_off()
        axs[r,1].text(.5,.5,"无高度数据",transform=axs[r,1].transAxes,ha="center",va="center",
                      color="#7d1f1f",bbox={"fc":"white","alpha":.92,"ec":"#b65c5c"})
    geo_height(axs[2,0],gdf,col,vmax,"（c）本文 GAMMA-MCF + SBAS 结果")
    x0,y0,x1,y1=detail;sub=gdf.cx[x0:x1,y0:y1];base=sub[sub[col].isna()];sol=sub[sub[col].notna()]
    base.plot(ax=axs[2,1],facecolor="#e8edf2",edgecolor="#6d8dad",lw=.15,rasterized=True)
    sol.plot(ax=axs[2,1],column=col,cmap="Spectral_r",vmin=0,vmax=vmax,edgecolor="#333",lw=.12,rasterized=True)
    axs[2,1].set_xlim(x0,x1);axs[2,1].set_ylim(y0,y1);axs[2,1].set_axis_off();axs[2,1].set_title("局部放大",fontsize=8)
    sm=mpl.cm.ScalarMappable(norm=Normalize(0,vmax),cmap="Spectral_r")
    cb=fig.colorbar(sm,ax=axs[2,:],fraction=.027,pad=.018);cb.set_label("建筑高度（m）")
    save(fig,out)


def fig49(gdf,out):
    x0,y0,x1,y1=117.18,39.115,117.225,39.145;sub=gdf.cx[x0:x1,y0:y1].to_crs(32650).copy();sub["area"]=sub.area;sub=sub.nlargest(1800,"area");h=sub["recommended_building_height_m"].fillna(10).clip(3,120);fig=plt.figure(figsize=(17*CM,11*CM));ax=fig.add_subplot(111,projection="3d",computed_zorder=False);bounds=sub.total_bounds
    xx,yy=np.meshgrid([bounds[0],bounds[2]],[bounds[1],bounds[3]]);ax.plot_surface(xx,yy,np.full_like(xx,-2.0),color="#253a59",shade=False,zorder=0)
    for geom,z,solved in zip(sub.geometry,h,sub["recommended_building_height_m"].notna()):
        geoms=list(geom.geoms) if geom.geom_type=="MultiPolygon" else [geom]
        for p in geoms:
            xy=np.asarray(p.exterior.coords)[::max(1,len(p.exterior.coords)//40)];top=[(x,y,float(z)) for x,y in xy];ax.add_collection3d(Poly3DCollection([top],facecolor="#ee8b2d" if solved else "#eeeeee",edgecolor="#555",linewidth=.08,zorder=4));
            sides=[[(xy[i,0],xy[i,1],0),(xy[i+1,0],xy[i+1,1],0),(xy[i+1,0],xy[i+1,1],float(z)),(xy[i,0],xy[i,1],float(z))] for i in range(len(xy)-1)];ax.add_collection3d(Poly3DCollection(sides,facecolor="#b76724" if solved else "#aeb5bd",edgecolor="none",zorder=3))
    ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3]);ax.set_zlim(-2,120);ax.set_box_aspect((1.35,1,.32));ax.view_init(31,-62);ax.set_axis_off();ax.text2D(.02,.96,"灰色：低层建筑白模（统一示意高度）   橙色：SBAS 有解建筑",transform=ax.transAxes,fontsize=7);save(fig,out)


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path("zjc/strict_reproduction"));a=p.parse_args();r=a.root;o=r/"results/paper_figure_reproduction";amp=amp_image(r/"work/amplitude/20231007.mli");search=np.load(r/"work/islands/independent_expanded_search_points.npz");quality=np.load(r/"work/islands/independent_expanded_quality_metrics.npz");pair="20240514_20240605.npz";obs=np.load(r/"work/pair_observations_independent_expanded_48"/pair);old=np.load(r/"work/unwrapped_original_expanded_48"/pair);ind=np.load(r/"work/unwrapped_independent_expanded_48"/pair);good=np.load(r/"results/paper_strict/pixel_height_independent_mcf_fixed_far_48.npz");bad=np.load(r/"results/paper_strict/pixel_height_independent_mcf_first_pixel_48.npz");gdf=gpd.read_file(r/"results/paper_strict/building_height_final.gpkg",layer="building_height")
    fig31(o/"图3.1_技术路线图.svg");fig32(o/"图3.2_建筑轮廓坐标转换示意图.svg");fig33(o/"图3.3_DBSCAN聚类示意图.svg");fig34(amp,o/"图3.4_实验区位置与雷达覆盖.svg");baseline(r/"work/baselines/paper_result_48.bperp",o/"图3.5_时空基线网络.svg");fig36(gdf,r/"work/buildings/floor_primary.rdc",o/"图3.6_建筑轮廓坐标转换结果.svg");fig37(amp,search,quality,o/"图3.7_独立建筑区域提取结果.svg");fig38(amp,obs,old,ind,o/"图3.8_建筑区域独立相位解缠结果.svg");one,two=choose_uids(obs);local_case(obs,old,ind,(one,),o/"图3.9_单体建筑相位解缠对比.svg","单体建筑相位解缠对比");local_case(obs,old,ind,two,o/"图3.10_相连建筑相位解缠对比.svg","相连建筑相位解缠对比");fig41(o/"图4.1_建筑物高程反演与三维重建流程.svg");fig42(amp,good,bad,o/"图4.2_不同解缠策略的建筑高程反演结果.svg");fig48(gdf,o/"图4.8_与光学建筑高度产品对比.svg");fig49(gdf,o/"图4.9_天津中心城区三维重建结果.svg");print(f"已输出 {len(list(o.glob('*.svg')))} 张 SVG：{o}")


if __name__ == "__main__": main()
