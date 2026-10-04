from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import BoundaryNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "section35"
NPZ = OUT / "prd130_v2_event_frequency.npz"
GEO = OUT / "static_geo.npz"
EVENTS = OUT / "prd130_v2_event_stats.csv"
SUMMARY = OUT / "prd130_v2_threshold_summary.csv"

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 20.0, "axes.titlesize": 21.0, "axes.labelsize": 20.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 1.15, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.labelsize": 17.0, "ytick.labelsize": 17.0,
})

def halo():
    return [pe.withStroke(linewidth=2.4, foreground="white", alpha=.95)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ); g = np.load(GEO)
    lat, lon = z["lat"], z["lon"]
    land = g["landmask"] > 0.5
    thresholds = z["thresholds"]
    j25 = int(np.argmin(np.abs(thresholds - 25)))
    delta = z["delta_frequency_pp"][j25]
    fe = z["freq_era5"][j25] * 100
    fp = z["freq_phytopo"][j25] * 100
    events = pd.read_csv(EVENTS)
    summ = pd.read_csv(SUMMARY)
    n = int(z["n_events"][0])

    extent = (110.6, 118.0, 19.0, 25.0)
    valid = np.isfinite(delta) & (lon>=extent[0]) & (lon<=extent[1]) & (lat>=extent[2]) & (lat<=extent[3])
    plot_delta = np.ma.masked_where(~valid | (delta < 0.43), delta)

    fig = plt.figure(figsize=(12.5, 8.4), facecolor="white")
    ax = fig.add_axes([0.070, 0.155, 0.640, 0.795])
    ax.set_facecolor("#eaf2f5")

    # Pure-vector map layer. Filled contours and isolines are calculated from
    # the same array and share the 5/10/15 boundaries, so their geometry is
    # identical in the PDF and remains editable in Illustrator.
    levels = np.arange(0, 18, 1)
    cmap = mpl.colormaps["YlOrRd"]
    norm = BoundaryNorm(levels, cmap.N, clip=True)
    ax.contourf(lon, lat, land.astype(float), levels=[.5, 1.5],
                colors=["#ecebe1"], antialiased=False, zorder=0)
    ax.contourf(lon, lat, plot_delta, levels=levels, cmap=cmap, norm=norm,
                extend="max", antialiased=False, zorder=2)
    pm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); pm.set_array([])
    ax.contour(lon, lat, land.astype(float), levels=[.5], colors="#31434d", linewidths=.62, zorder=4)
    cs = ax.contour(lon, lat, np.where(valid, delta, np.nan), levels=[5,10,15],
                    colors=["#ad5b2d","#7f2e20","#5a1414"], linewidths=[.55,.8,1.05], zorder=3)
    ax.clabel(cs, fmt=lambda x:f"+{x:.0f}", inline=True, fontsize=17.0)

    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_:f"{x:.0f}°E"))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y,_:f"{y:.0f}°N"))
    ax.set_xlabel(""); ax.set_ylabel("Latitude")
    ax.grid(color="#aab7bd", lw=.42, alpha=.38, zorder=1)
    for s in ax.spines.values(): s.set_color("#50616b")

    cities = {
        "Guangzhou":(113.264,23.129), "Shenzhen":(114.057,22.544),
        "Hong Kong":(114.169,22.319), "Macao":(113.543,22.199),
        "Shantou":(116.682,23.354), "Haikou":(110.199,20.045)
    }
    for name,(x,y) in cities.items():
        if extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]:
            ax.plot(x,y,marker="o",ms=2.7,mec="#17242b",mfc="white",mew=.7,zorder=7)
            dx = .08 if name not in {"Shenzhen","Hong Kong"} else .10
            dy = .09 if name != "Hong Kong" else -.18
            ax.text(x+dx,y+dy,name,fontsize=18.0,color="#1f3038",path_effects=halo(),zorder=8)

    idx = np.nanargmax(np.where(valid,delta,np.nan))
    hx,hy,hv = lon.ravel()[idx],lat.ravel()[idx],delta.ravel()[idx]
    ax.scatter(hx,hy,s=44,marker="*",c="#702018",edgecolor="white",linewidth=.65,zorder=8)
    ax.annotate(f"maximum +{hv:.1f} pp",(hx,hy),xytext=(18,14),textcoords="offset points",
                fontsize=17.0,color="#6d201a",fontweight="bold",path_effects=halo(),
                arrowprops=dict(arrowstyle="-",lw=1.0,color="#6d201a"),zorder=9)

    cax = fig.add_axes([0.10,0.070,0.535,0.025])
    cb = fig.colorbar(pm,cax=cax,orientation="horizontal",ticks=np.arange(0,18,2),extend="max")
    cb.set_label("Increase in 25 m/s event-exceedance frequency (percentage points)",labelpad=4)

    # Integrated threshold summary (no panel label)
    axi = fig.add_axes([0.770,0.575,0.210,0.355],facecolor="white")
    x = summ.threshold_ms.values
    axi.plot(x,summ.era5_events.values,"o-",color="#3b78b4",lw=1.8,ms=5,label="ERA5")
    axi.plot(x,summ.phytopo_events.values,"o-",color="#df7f18",lw=1.8,ms=5,label="PhyTopo")
    axi.fill_between(x,summ.era5_events.values,summ.phytopo_events.values,color="#efb264",alpha=.18)
    for xx,ce,cr in zip(x,summ.era5_events,summ.phytopo_events):
        axi.text(xx,cr+3,f"{int(cr)}",ha="center",va="bottom",fontsize=16.5,color="#9a4f0d")
        axi.text(xx,(ce+cr)/2,f"+{int(cr-ce)}",ha="center",va="center",fontsize=16.0,
                 color="#6f4b25",fontweight="bold",
                 bbox=dict(boxstyle="round,pad=.13",fc="white",ec="none",alpha=.78))
    axi.set_title("Threshold-dependent event identification",fontweight="bold",loc="left",pad=7,fontsize=19)
    axi.set_xlabel("Event-maximum threshold (m/s)",fontsize=16)
    axi.set_ylabel("Number of events",fontsize=17)
    axi.tick_params(labelsize=15)
    axi.set_xticks(x); axi.set_ylim(0,116); axi.grid(axis="y",color="#d9e0e4",lw=.6)
    axi.legend(frameon=False,loc="upper right",fontsize=15.5,ncol=2,handlelength=1.15,
               columnspacing=.8)
    axi.spines[["top","right"]].set_visible(False)

    # Paired increment distribution across all events. This directly shows
    # prevalence and magnitude without making the modest correlation change
    # carry the catalogue-scale result.
    axs = fig.add_axes([0.770,0.135,0.210,0.335], facecolor="white")
    inc = np.sort(events["increment_ms"].to_numpy())
    rank = np.arange(1, len(inc)+1)
    colors = np.where(inc >= 0, "#df7f18", "#3b78b4")
    axs.bar(rank, inc, width=1.0, color=colors, edgecolor="none")
    med = float(np.median(inc))
    npos = int(np.sum(inc > 0)); nge5 = int(np.sum(inc >= 5)); nge10 = int(np.sum(inc >= 10))
    axs.axhline(0,color="#36454f",lw=1.05)
    axs.axhline(med,color="#6b3810",lw=1.25,ls="--")
    axs.axhline(5,color="#9b8b7c",lw=.75,ls=":")
    axs.axhline(10,color="#9b8b7c",lw=.75,ls=":")
    axs.text(.04,.96,f"{npos}/{len(inc)} increased ({100*npos/len(inc):.1f}%)",
             transform=axs.transAxes,ha="left",va="top",fontsize=15.2,fontweight="bold",color="#8d4809")
    axs.text(.04,.86,f"Median +{med:.2f} m/s",transform=axs.transAxes,
             ha="left",va="top",fontsize=14.7,color="#5a4634")
    axs.text(.04,.76,f">= 5 m/s: {nge5}/{len(inc)}",transform=axs.transAxes,
             ha="left",va="top",fontsize=13.7,color="#5a4634")
    axs.text(.04,.67,f">= 10 m/s: {nge10}/{len(inc)}",transform=axs.transAxes,
             ha="left",va="top",fontsize=13.7,color="#5a4634")
    axs.set_xlim(.5,len(inc)+.5); axs.set_ylim(min(-2.5,inc.min()-.5),20.5)
    axs.set_title("Event-wide maximum-wind increments",fontweight="bold",loc="left",pad=7,fontsize=19)
    axs.set_xlabel("Typhoons ranked by increment",fontsize=15.5)
    axs.set_ylabel("PhyTopo - ERA5 (m/s)",fontsize=16.0)
    axs.set_xticks([1,30,60,90,117]); axs.tick_params(labelsize=14.5)
    axs.grid(axis="y",color="#d9e0e4",lw=.6,alpha=.9)
    axs.spines[["top","right"]].set_visible(False)

    png = OUT / "Figure10_threshold_dependent_reassessment.png"
    pdf = OUT / "Figure10_threshold_dependent_reassessment_pure_vector.pdf"
    fig.savefig(png,dpi=400,bbox_inches="tight",facecolor="white")
    fig.savefig(pdf,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    print(png); print(pdf)

if __name__ == "__main__": main()
