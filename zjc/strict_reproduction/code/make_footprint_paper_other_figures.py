#!/usr/bin/env python3
"""Reproduce the non-statistical figures of the Footprint-Constrained paper.

Outputs Figures 1, 2, 3, 5, 6, 7 and 13 as separate Chinese SVGs.  Figures
8-12 are intentionally excluded as error-statistics figures.  Missing external
CNBH/GBA rasters and the missing full-scene global-MCF product are never
fabricated or silently relabelled.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_paper_layout_figures as base  # noqa: E402


W, H = 10_000, 7_000
CM = 1 / 2.54
OUT_BBOX = (117.15, 39.10, 117.24, 39.16)
ZOOM_BBOX = (117.195, 39.118, 117.215, 39.137)

mpl.rcParams.update({
    "font.family": "Noto Sans CJK SC",
    "font.sans-serif": ["Noto Sans CJK SC"],
    "svg.fonttype": "none",
    "axes.unicode_minus": False,
    "font.size": 7,
})


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight", dpi=300)
    plt.close(fig)


def module_box(ax, xy, wh, title):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                       fc="#fff8f1", ec="#111", lw=1.0, ls=(0, (5, 3)))
    ax.add_patch(p)
    ax.text(x+w/2, y+h-.022, title, ha="center", va="top", fontsize=6.8,
            linespacing=1.12)
    return p


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10,
                                 lw=1.0, color="#111"))


def fig1_workflow(amp, gdf, search, obs, out):
    """2x2 workflow matching the paper's schematic-led layout."""
    fig, ax = plt.subplots(figsize=(18.3*CM, 9.2*CM))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    modules = [
        ((.02,.54),(.43,.42),"（1）InSAR 数据预处理\n与干涉图生成"),
        ((.02,.04),(.43,.42),"（2）矢量雷达编码\n与建筑掩膜构建"),
        ((.55,.54),(.43,.42),"（3）掩膜约束的\n建筑独立相位解缠"),
        ((.55,.04),(.43,.42),"（4）分层基线建筑高度估计\n与三维重建"),
    ]
    for xy,wh,t in modules: module_box(ax,xy,wh,t)

    # Module 1: a compact stack made from the real SAR amplitude plate.
    crop = amp[700:1250, 850:1700]
    for k in range(5):
        x0=.095+.025*k; y0=.575+.014*k
        ax.imshow(crop, cmap="gray", extent=(x0,x0+.235,y0,y0+.18),
                  transform=ax.transData, interpolation="nearest", rasterized=True,
                  zorder=3+k)

    # Module 2: actual footprint and radar-support miniatures.
    left = ax.inset_axes([.055,.085,.17,.22]); right=ax.inset_axes([.27,.085,.15,.22])
    x0,y0,x1,y1=ZOOM_BBOX
    geo=gdf.cx[x0:x1,y0:y1]
    geo.plot(ax=left,facecolor="#b74646",edgecolor="white",lw=.15,rasterized=True)
    left.set_axis_off();left.text(.5,.03,"二维建筑轮廓",transform=left.transAxes,ha="center",va="bottom",fontsize=5.4,bbox={"fc":"white","alpha":.78,"ec":"none","pad":.6})
    m=(search["col"]>5200)&(search["col"]<6500)&(search["row"]>3300)&(search["row"]<4300)
    right.scatter(search["col"][m],search["row"][m],s=.15,c="white",lw=0,rasterized=True)
    right.set_facecolor("#262d2f");right.set_xlim(5200,6500);right.set_ylim(4300,3300);right.set_axis_off();right.text(.5,.03,"雷达建筑掩膜",transform=right.transAxes,ha="center",va="bottom",fontsize=5.4,bbox={"fc":"black","alpha":.62,"ec":"none","pad":.6},color="white")
    ax.text(.247,.22,"R-D\n编码",ha="center",va="center",fontsize=6)
    arrow(ax,(.225,.22),(.27,.22))

    # Module 3: real wrapped phases in several actual building islands.
    phase_ax=ax.inset_axes([.59,.575,.34,.22]);phase_ax.set_facecolor("#f1efe8")
    m=(obs["col"]>5200)&(obs["col"]<6800)&(obs["row"]>3300)&(obs["row"]<4300)
    phase_ax.scatter(obs["col"][m],obs["row"][m],c=obs["filtered_wrapped_phase_rad"][m],
                     cmap="hsv",vmin=-np.pi,vmax=np.pi,s=.20,lw=0,rasterized=True)
    phase_ax.set_xlim(5200,6800);phase_ax.set_ylim(4300,3300);phase_ax.set_axis_off()
    phase_ax.text(.02,.96,"建筑边界阻断跨域误差传播",transform=phase_ax.transAxes,
                  ha="left",va="top",fontsize=5.8,bbox={"fc":"white","alpha":.78,"ec":"none"})

    # Module 4: compact oblique block model (actual solved heights).
    solved=gdf[gdf["recommended_building_height_m"].notna()].cx[x0:x1,y0:y1]
    model=ax.inset_axes([.60,.075,.32,.22]);model.set_facecolor("#9aa6b8")
    model.plot([0,.25,.45,.68,1],[.25,.38,.30,.53,.48],c="#174e73",lw=5)
    if len(solved):
        hh=solved["recommended_building_height_m"].clip(0,120).to_numpy()
        xx=np.linspace(.06,.94,len(hh)); model.bar(xx,.10+.58*hh/120,width=.012,bottom=.18,
                                                  color="#f28e2b",edgecolor="none")
    model.set_xlim(0,1);model.set_ylim(0,1);model.set_axis_off();model.text(.5,.04,"三维建筑模型",ha="center",fontsize=6)

    arrow(ax,(.235,.54),(.235,.47));arrow(ax,(.45,.25),(.55,.75));arrow(ax,(.765,.54),(.765,.47))
    save(fig,out)


def fig6_height_comparison(amp, proposed, control, out):
    crop=(4300,7300,3300,5300)
    fig,axs=plt.subplots(1,2,figsize=(18.3*CM,7.0*CM),gridspec_kw={"wspace":.36})
    specs=[(proposed,"（a）建筑轮廓约束的独立反演",Normalize(0,220)),
           (control,"（b）覆盖式 MCF 对照（非整幅全局 MCF）",Normalize(-60,220))]
    scat=[]
    for ax,(d,title,norm) in zip(axs,specs):
        x=d["col"]-crop[0];y=d["row"]-crop[2];m=(x>=0)&(x<3000)&(y>=0)&(y<2000)
        v=d["dem_error_or_height_above_anchor_m"]
        ax.set_facecolor("white")
        sc=ax.scatter(x[m],y[m],c=v[m],s=.62,cmap="jet",norm=norm,lw=0,rasterized=True)
        scat.append(sc);ax.set_xlim(0,3000);ax.set_ylim(2000,0);ax.set_aspect("equal");ax.set_xlabel("距离向像元")
        ax.text(.018,.975,title,transform=ax.transAxes,ha="left",va="top",fontsize=7,
                bbox={"fc":"white","alpha":.82,"ec":"none","pad":1.2})
    axs[0].set_ylabel("方位向像元"); axs[1].set_ylabel("")
    for ax,sc in zip(axs,scat):
        cb=fig.colorbar(sc,ax=ax,fraction=.032,pad=.018);cb.set_label("相对高差（m）")
    save(fig,out)


def fig7_3d(gdf,out):
    """Paper-like wide oblique 3-D city plate without excess page whitespace."""
    x0,y0,x1,y1=117.18,39.115,117.225,39.145
    sub=gdf.cx[x0:x1,y0:y1].to_crs(32650).copy();sub["area"]=sub.area;sub=sub.nlargest(1800,"area")
    heights=sub["recommended_building_height_m"].fillna(10).clip(3,120)
    fig=plt.figure(figsize=(18.3*CM,7.2*CM));ax=fig.add_axes([0,0,1,1],projection="3d",computed_zorder=False)
    b=sub.total_bounds;xx,yy=np.meshgrid([b[0],b[2]],[b[1],b[3]]);ax.plot_surface(xx,yy,np.full_like(xx,-2),color="#708097",shade=False,zorder=0)
    for geom,z,solved in zip(sub.geometry,heights,sub["recommended_building_height_m"].notna()):
        parts=list(geom.geoms) if geom.geom_type=="MultiPolygon" else [geom]
        for poly in parts:
            xy=np.asarray(poly.exterior.coords)[::max(1,len(poly.exterior.coords)//36)]
            top=[(x,y,float(z)) for x,y in xy]
            ax.add_collection3d(Poly3DCollection([top],facecolor="#f39a2d" if solved else "#dbe3ea",edgecolor="#77818a",linewidth=.05,zorder=4))
            sides=[[(xy[i,0],xy[i,1],0),(xy[i+1,0],xy[i+1,1],0),(xy[i+1,0],xy[i+1,1],float(z)),(xy[i,0],xy[i,1],float(z))] for i in range(len(xy)-1)]
            ax.add_collection3d(Poly3DCollection(sides,facecolor="#bf6f20" if solved else "#aeb9c4",edgecolor="none",zorder=3))
    ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3]);ax.set_zlim(-2,120);ax.set_box_aspect((1.55,1,.30));ax.view_init(28,-65);ax.set_axis_off();save(fig,out)


def map_panel(ax, gdf, available, title, zoom=False):
    bbox=ZOOM_BBOX if zoom else OUT_BBOX
    x0,y0,x1,y1=bbox;sub=gdf.cx[x0:x1,y0:y1]
    if available:
        baseg=sub[sub["recommended_building_height_m"].isna()]
        sol=sub[sub["recommended_building_height_m"].notna()]
        baseg.plot(ax=ax,facecolor="#edf2f7",edgecolor="#b7cbe0",lw=.05,rasterized=True)
        sol.plot(ax=ax,column="recommended_building_height_m",cmap="Spectral_r",vmin=0,vmax=150,
                 edgecolor="none",rasterized=True)
    else:
        sub.plot(ax=ax,facecolor="#eef1f4",edgecolor="#a8b8c7",lw=.06,rasterized=True)
        ax.text(.5,.5,"外部高度产品未提供\n仅复现论文版式",transform=ax.transAxes,ha="center",va="center",fontsize=7,
                bbox={"fc":"white","alpha":.90,"ec":"#c44","pad":2})
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect(1/np.cos(np.deg2rad(39.13)))
    if zoom:ax.set_axis_off()
    else:
        ax.set_xlabel("经度");ax.set_ylabel("纬度");ax.set_title(title,fontsize=7,pad=2)
        zx0,zy0,zx1,zy1=ZOOM_BBOX;ax.add_patch(Rectangle((zx0,zy0),zx1-zx0,zy1-zy0,fill=False,ec="#e31a1c",lw=.9))


def fig13_optical_comparison(gdf,out):
    fig,axs=plt.subplots(2,3,figsize=(18.3*CM,10.5*CM),gridspec_kw={"height_ratios":[1.05,.75],"hspace":.10,"wspace":.14})
    titles=["（a）CNBH-10 m","（b）GBA","（c）本文方法"]
    for j,title in enumerate(titles):
        available=j==2
        map_panel(axs[0,j],gdf,available,title,False);map_panel(axs[1,j],gdf,available,"",True)
        axs[1,j].add_patch(Rectangle((ZOOM_BBOX[0]+.002,ZOOM_BBOX[1]+.003),.004,.004,fill=False,ec="#e31a1c",lw=.9))
    sm=mpl.cm.ScalarMappable(norm=Normalize(0,150),cmap="Spectral_r")
    cb=fig.colorbar(sm,ax=axs[:,2],fraction=.035,pad=.015);cb.set_label("建筑高度（m）")
    save(fig,out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,default=Path("zjc/strict_reproduction"));args=parser.parse_args();r=args.root;o=r/"results/footprint_constrained_paper"
    amp=base.amp_image(r/"work/amplitude/20231007.mli")
    search=np.load(r/"work/islands/independent_expanded_search_points.npz")
    quality=np.load(r/"work/islands/independent_expanded_quality_metrics.npz")
    obs=np.load(r/"work/pair_observations_independent_expanded_48/20240514_20240605.npz")
    proposed=np.load(r/"results/paper_strict/pixel_height_independent_mcf_fixed_far_48.npz")
    control=np.load(r/"results/paper_strict/pixel_height_mcf_fixed_far_48.npz")
    gdf=gpd.read_file(r/"results/paper_strict/building_height_final.gpkg",layer="building_height")
    fig1_workflow(amp,gdf,search,obs,o/"图1_轮廓约束建筑高度反演总体流程.svg")
    base.fig34(amp,o/"图2_天津中心城区研究区概览.svg")
    base.baseline(r/"work/baselines/paper_result_48.bperp",o/"图3_时空基线分布.svg")
    base.fig37(amp,search,quality,o/"图5_研究区建筑孤岛提取结果.svg")
    fig6_height_comparison(amp,proposed,control,o/"图6_不同解缠策略的建筑高度反演对比.svg")
    fig7_3d(gdf,o/"图7_天津中心城区三维重建.svg")
    fig13_optical_comparison(gdf,o/"图13_建筑高度图与光学产品版式对比.svg")
    print(f"完成：{len(list(o.glob('*.svg')))} 张 SVG（含已完成 Fig.4）")


if __name__=="__main__":main()
