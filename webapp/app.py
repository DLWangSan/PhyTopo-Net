from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import xarray as xr
from plotly.subplots import make_subplots
from scipy.interpolate import RegularGridInterpolator

import predict_full_field as pf
from model import PhyTopoNet
from visualization import predict_full_field_blend_adv as pfa


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "experiments" / "phy_topo_stage3_semisup" / "best_model.pth"
GEO_PATH = ROOT / "wrf" / "data" / "geo_em.d03.nc"
PEAK_MODEL_PATH = ROOT / "pressure_peak_model.npz"
EXAMPLE_ERA5 = ROOT / "examples" / "Mangkhut_2018_ERA5_full_event.nc"
EXAMPLE_TRACK = ROOT / "examples" / "Mangkhut_2018_track_pressure.csv"
BOUNDARY_PATH = ROOT / "assets" / "prd_boundaries.geojson"

BLUE = "#2C73B9"
ORANGE = "#E48118"
GREEN = "#31845B"
NAVY = "#173A5E"


try:
    import spaces
    # The preview finishes in seconds; keep the declared reservation below the
    # public ZeroGPU per-call ceiling so the Space remains usable without PRO.
    gpu_task = spaces.GPU(duration=90)
except Exception:
    def gpu_task(func):
        return func


@lru_cache(maxsize=2)
def load_runtime(requested_device="cpu"):
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    residual = checkpoint.get("residual", False) if isinstance(checkpoint, dict) else False
    model = PhyTopoNet(residual=residual)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    static, lat, lon = pf.load_static_data(str(GEO_PATH))
    return model, device, static, lat, lon


def normalize_file(value, fallback):
    if value is None:
        return Path(fallback)
    if isinstance(value, str):
        return Path(value)
    if hasattr(value, "name"):
        return Path(value.name)
    return Path(value)


def input_times(path):
    with xr.open_dataset(path) as ds:
        name = "valid_time" if "valid_time" in ds.coords else "time"
        if name not in ds.coords:
            raise ValueError("The ERA5 file must contain a valid_time or time coordinate.")
        return name, list(pd.to_datetime(ds[name].values).round("h"))


def time_labels_and_default(path):
    _, times = input_times(path)
    labels = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC") for t in times]
    if not labels:
        raise ValueError("The ERA5 file contains no timestamps.")
    return labels, labels[len(labels) // 2]


def timestamp_choices(value):
    path = normalize_file(value, EXAMPLE_ERA5)
    labels, default_time = time_labels_and_default(path)
    return gr.update(choices=labels, value=default_time)


def subset_times(times, choice, selected_label):
    count = {"Single timestamp": 1, "6-hour animation": 6,
             "24-hour animation": 24}[choice]
    labels = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC") for t in times]
    center = labels.index(selected_label) if selected_label in labels else len(times) // 2
    start = max(0, min(center - count // 2, len(times) - count))
    return list(times[start:start + min(count, len(times))])


def run_direct(era5_path, selected_times, requested_device="cpu"):
    model, device, static, wrf_lat, wrf_lon = load_runtime(requested_device)
    pred_u, pred_v, pred_p, era_u, era_v = [], [], [], [], []
    completed = []
    for timestamp in selected_times:
        loaded = pf.load_era5_for_time(
            str(era5_path), timestamp, wrf_lat, wrf_lon, return_raw=False,
        )
        if loaded is None:
            continue
        required = ("U10", "V10", "SLP", "T2", "SST", "BLH", "FSR")
        missing = [name for name in required if name not in loaded]
        if missing:
            raise ValueError("Missing required ERA5 variables after mapping: " + ", ".join(missing))
        predicted = pfa.predict_full_field_advanced(
            model, loaded, static, device, patch_size=64, stride=16,
            blend="hann", dual_grid=True, triple_grid=True,
            norm_uv_smooth_sigma=0.35,
            batch_size=8 if device == "cuda" else 1,
        )
        physical = np.zeros((*predicted.shape[:2], 4), dtype=np.float32)
        for channel, name in enumerate(("U10", "V10", "SLP", "T2")):
            physical[..., channel] = pf.denormalize(predicted[..., channel], name)
        pred_u.append(physical[..., 0])
        pred_v.append(physical[..., 1])
        pred_p.append(physical[..., 2])
        era_u.append(np.asarray(loaded["U10"], np.float32))
        era_v.append(np.asarray(loaded["V10"], np.float32))
        completed.append(pd.Timestamp(timestamp))
    if not completed:
        raise ValueError("No valid timestamps could be processed.")
    return {
        "time": pd.DatetimeIndex(completed), "lat": np.asarray(wrf_lat),
        "lon": np.asarray(wrf_lon), "u": np.stack(pred_u), "v": np.stack(pred_v),
        "msl": np.stack(pred_p), "era_u": np.stack(era_u), "era_v": np.stack(era_v),
        "hgt": static[..., 0], "landmask": static[..., 4],
    }


def save_direct(result):
    path = Path(tempfile.mkdtemp(prefix="phytopo_")) / "phytopo_reconstruction.nc"
    ds = xr.Dataset(
        data_vars={
            "u10": (("time", "south_north", "west_east"), result["u"],
                    {"units": "m s-1", "long_name": "PhyTopo-Net 10-m eastward wind"}),
            "v10": (("time", "south_north", "west_east"), result["v"],
                    {"units": "m s-1", "long_name": "PhyTopo-Net 10-m northward wind"}),
            "msl": (("time", "south_north", "west_east"), result["msl"],
                    {"units": "hPa", "long_name": "PhyTopo-Net mean sea-level pressure"}),
        },
        coords={
            "time": result["time"],
            "XLAT": (("south_north", "west_east"), result["lat"]),
            "XLONG": (("south_north", "west_east"), result["lon"]),
        },
        attrs={
            "title": "PhyTopo-Net direct reconstruction",
            "inference": (
                "64 x 64 patches; stride 16; Hann blending; three offset grids; "
                "normalized U/V Gaussian smoothing sigma=0.35"
            ),
            "static_domain": "Pearl River Delta 3-km WRF domain",
        },
    )
    ds.to_netcdf(path, encoding={name: {"zlib": True, "complevel": 4}
                                 for name in ds.data_vars})
    ds.close()
    return path


@lru_cache(maxsize=1)
def boundary_lines():
    lines = {"province": ([], []), "prefecture": ([], [])}
    if not BOUNDARY_PATH.exists():
        return lines

    def append_geometry(geometry, target):
        kind = geometry.get("type", "")
        coordinates = geometry.get("coordinates", [])
        if kind == "Polygon":
            groups = coordinates
        elif kind == "MultiPolygon":
            groups = [ring for polygon in coordinates for ring in polygon]
        elif kind == "LineString":
            groups = [coordinates]
        elif kind == "MultiLineString":
            groups = coordinates
        else:
            groups = []
        for group in groups:
            if not group:
                continue
            target[0].extend(point[0] for point in group)
            target[1].extend(point[1] for point in group)
            target[0].append(None)
            target[1].append(None)

    content = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    for feature in content.get("features", []):
        layer = feature.get("properties", {}).get("layer", "prefecture")
        append_geometry(feature.get("geometry", {}), lines.setdefault(layer, ([], [])))
    return lines


def make_map(result):
    era_speed = np.hypot(result["era_u"], result["era_v"])
    phy_speed = np.hypot(result["u"], result["v"])
    vmax = max(10.0, float(np.nanpercentile(np.r_[era_speed.ravel(), phy_speed.ravel()], 99.7)))
    step = 3
    x = result["lon"][0, ::step]
    y = result["lat"][::step, 0]
    geographic_y_scale = 1.0 / np.cos(np.deg2rad(float(np.nanmean(y))))
    labels = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M UTC") for t in result["time"]]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.035,
                        subplot_titles=("ERA5 interpolation", "PhyTopo reconstruction"))
    heatmap = dict(x=x, y=y, coloraxis="coloraxis", hovertemplate=(
        "Lon %{x:.2f}°E<br>Lat %{y:.2f}°N<br>Wind %{z:.2f} m/s<extra></extra>"
    ))
    fig.add_trace(go.Heatmap(z=era_speed[0, ::step, ::step], **heatmap), row=1, col=1)
    fig.add_trace(go.Heatmap(z=phy_speed[0, ::step, ::step], **heatmap), row=1, col=2)
    terrain = np.where(result["landmask"] > 0.5, result["hgt"], np.nan)[::step, ::step]
    for col in (1, 2):
        fig.add_trace(go.Contour(
            z=terrain, x=x, y=y, contours={"coloring": "none", "showlabels": False},
            ncontours=10, line={"color": "rgba(73,94,103,0.32)", "width": 0.55},
            showscale=False, hoverinfo="skip", showlegend=False,
        ), row=1, col=col)
    for layer, (bx, by) in boundary_lines().items():
        style = dict(color="#163A59" if layer == "province" else "rgba(22,58,89,0.55)",
                     width=1.15 if layer == "province" else 0.65)
        for col in (1, 2):
            fig.add_trace(go.Scattergl(x=bx, y=by, mode="lines", line=style,
                                       hoverinfo="skip", showlegend=False), row=1, col=col)
    frames = []
    for idx, label in enumerate(labels):
        frames.append(go.Frame(
            name=label,
            data=[go.Heatmap(z=era_speed[idx, ::step, ::step], x=x, y=y, coloraxis="coloraxis"),
                  go.Heatmap(z=phy_speed[idx, ::step, ::step], x=x, y=y, coloraxis="coloraxis")],
            traces=[0, 1],
            layout=go.Layout(title_text=label),
        ))
    fig.frames = frames
    slider_steps = [{"args": [[label], {"frame": {"duration": 320, "redraw": True},
                                       "mode": "immediate", "transition": {"duration": 0}}],
                     "label": pd.Timestamp(t).strftime("%m-%d %H:%M"),
                     "method": "animate"}
                    for label, t in zip(labels, result["time"])]
    fig.update_layout(
        title={"text": labels[0], "x": 0.5, "xanchor": "center", "font": {"size": 16}},
        margin={"l": 48, "r": 30, "t": 62, "b": 86}, height=465,
        paper_bgcolor="white", plot_bgcolor="#EEF5F8", font={"family": "Arial", "color": "#263744"},
        coloraxis={"colorscale": "Turbo", "cmin": 0, "cmax": vmax,
                   "colorbar": {"title": "10-m wind<br>(m/s)", "thickness": 14, "len": 0.72}},
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0.0, "y": -0.15,
                      "showactive": False, "buttons": [
                          {"label": "▶ Play", "method": "animate",
                           "args": [None, {"frame": {"duration": 650, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 0}}]},
                          {"label": "❚❚ Pause", "method": "animate",
                           "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                             "mode": "immediate", "transition": {"duration": 0}}]},
                      ]}],
        sliders=[{"active": 0, "currentvalue": {"prefix": "UTC  "}, "pad": {"t": 46},
                  "steps": slider_steps}],
    )
    for axis in ("xaxis", "xaxis2"):
        fig.layout[axis].update(title="Longitude (°E)", range=[float(np.nanmin(x)), float(np.nanmax(x))],
                                showgrid=False, zeroline=False, constrain="domain")
    for axis, anchor in (("yaxis", "x"), ("yaxis2", "x2")):
        fig.layout[axis].update(title="Latitude (°N)" if axis == "yaxis" else "",
                                range=[float(np.nanmin(y)), float(np.nanmax(y))],
                                showgrid=False, zeroline=False, constrain="domain",
                                scaleanchor=anchor, scaleratio=geographic_y_scale)
    return fig


def make_peak_series(result):
    era = np.nanmax(np.hypot(result["era_u"], result["era_v"]), axis=(1, 2))
    phy = np.nanmax(np.hypot(result["u"], result["v"]), axis=(1, 2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["time"], y=era, name="ERA5-Interp.", mode="lines+markers",
                             line={"color": BLUE, "width": 2}, marker={"size": 5}))
    fig.add_trace(go.Scatter(x=result["time"], y=phy, name="PhyTopo", mode="lines+markers",
                             line={"color": ORANGE, "width": 2.4}, marker={"size": 5}))
    fig.update_layout(height=255, margin={"l": 48, "r": 14, "t": 22, "b": 42},
                      paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
                      legend={"orientation": "h", "y": 1.14, "x": 0},
                      font={"family": "Arial", "size": 11, "color": "#263744"},
                      xaxis_title="UTC time", yaxis_title="Domain peak (m/s)")
    fig.update_xaxes(showgrid=True, gridcolor="#E6EDF2")
    fig.update_yaxes(showgrid=True, gridcolor="#E6EDF2", rangemode="tozero")
    return fig


def interpolate_track(track_path, times):
    track = pd.read_csv(track_path)
    required = {"time_utc", "latitude", "longitude", "central_pressure_hpa"}
    missing = required.difference(track.columns)
    if missing:
        raise ValueError("Track CSV is missing: " + ", ".join(sorted(missing)))
    source_time = pd.to_datetime(track["time_utc"], utc=True).astype("int64") / 1e9
    target_time = pd.to_datetime(times, utc=True).astype("int64") / 1e9
    order = np.argsort(source_time)
    values = {}
    for name in ("latitude", "longitude", "central_pressure_hpa"):
        values[name] = np.interp(target_time, source_time[order],
                                 track[name].to_numpy(float)[order])
    return values


def sample_regular(field, source_lat, source_lon, target_lat, target_lon):
    lat = np.asarray(source_lat, float)
    lon = np.asarray(source_lon, float)
    values = np.asarray(field, float)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        values = values[::-1, :]
    interp = RegularGridInterpolator((lat, lon), values, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([target_lat, target_lon]))


def nearest_curvilinear(field, grid_lat, grid_lon, target_lat, target_lon):
    out = []
    for lat, lon in zip(target_lat, target_lon):
        distance = (grid_lat - lat) ** 2 + ((grid_lon - lon) * np.cos(np.deg2rad(lat))) ** 2
        iy, ix = np.unravel_index(np.nanargmin(distance), distance.shape)
        out.append(field[..., iy, ix])
    return np.stack(out, axis=-1)


def pressure_station_peaks(result, era5_path, track_path):
    model = {key: value for key, value in np.load(PEAK_MODEL_PATH, allow_pickle=False).items()}
    keys = model["station_key"].astype(str)
    station_lat = model["station_lat"].astype(float)
    station_lon = model["station_lon"].astype(float)
    phy_speed = np.hypot(result["u"], result["v"])
    phy_series = nearest_curvilinear(phy_speed, result["lat"], result["lon"],
                                    station_lat, station_lon)

    with xr.open_dataset(era5_path) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        ds = ds.rename({tname: "time"}) if tname != "time" else ds
        ds["time"] = pd.to_datetime(ds.time.values).round("h")
        era_series = []
        for timestamp in result["time"]:
            selected = ds.sel(time=pd.Timestamp(timestamp), method="nearest")
            speed = np.hypot(selected["u10"].values, selected["v10"].values)
            era_series.append(sample_regular(speed, selected.latitude.values,
                                             selected.longitude.values,
                                             station_lat, station_lon))
    era_series = np.stack(era_series)
    track = interpolate_track(track_path, result["time"])
    dp_t = np.maximum(1010.0 - track["central_pressure_hpa"], 0.0)
    dx = (station_lon[None, :] - track["longitude"][:, None]) * 111.0 * np.cos(
        np.deg2rad((station_lat[None, :] + track["latitude"][:, None]) / 2.0)
    )
    dy = (station_lat[None, :] - track["latitude"][:, None]) * 111.0
    radius = np.hypot(dx, dy)

    phy = np.nanmax(phy_series, axis=0)
    era = np.nanmax(era_series, axis=0)
    dp = np.full_like(phy, float(np.nanmax(dp_t)))
    r = np.nanmin(radius, axis=0)
    core = np.exp(-r / 300.0)
    features = np.column_stack([
        phy, era, np.sqrt(dp), core, dp * core, phy * np.sqrt(dp), phy * core,
        np.maximum(phy - 10.0, 0.0), np.maximum(phy - 20.0, 0.0),
    ])
    standardized = (features - model["scaler_mean"]) / model["scaler_scale"]
    raw = float(model["model_intercept"]) + standardized @ model["global_coef"]
    raw += model["station_intercept_coef"]
    raw += model["station_speed_coef"] * phy
    raw += model["station_hinge_coef"] * np.maximum(phy - 15.0, 0.0)
    raw += model["station_core_coef"] * core
    spread = float(model["spread_mu_obs"]) + (
        raw - float(model["spread_mu_raw"])
    ) * float(model["spread_ratio"])
    activation = 1.0 / (1.0 + np.exp(-(
        spread - float(model["activation_threshold"])
    ) / float(model["activation_width"])))
    corrected = np.maximum(0.0, spread + float(model["tail_delta"]) * activation)
    table = pd.DataFrame({
        "station_id": keys, "latitude": station_lat, "longitude": station_lon,
        "ERA5_peak_m_s": era, "PhyTopo_peak_m_s": phy,
        "pressure_constrained_peak_m_s": corrected,
    })
    path = Path(tempfile.mkdtemp(prefix="phytopo_peak_")) / "pressure_constrained_station_peaks.csv"
    table.to_csv(path, index=False, float_format="%.3f")
    return table, path


def infer_impl(era5_file, track_file, duration, selected_timestamp, mode, requested_device):
    try:
        era5_path = normalize_file(era5_file, EXAMPLE_ERA5)
        track_path = normalize_file(track_file, EXAMPLE_TRACK)
        _, times = input_times(era5_path)
        selected = subset_times(times, duration, selected_timestamp)
        result = run_direct(era5_path, selected, requested_device=requested_device)
        direct_file = save_direct(result)
        figure = make_map(result)
        peak_series = make_peak_series(result)
        era_peak = float(np.nanmax(np.hypot(result["era_u"], result["era_v"])))
        phy_peak = float(np.nanmax(np.hypot(result["u"], result["v"])))
        summary = pd.DataFrame({
            "Product": ["ERA5-Interp.", "PhyTopo"],
            "Peak (m/s)": [era_peak, phy_peak],
            "Increment over paired ERA5 (m/s)": [0.0, phy_peak - era_peak],
        })
        peak_file = None
        pressure_ready = duration == "24-hour animation" and len(times) >= 24
        if mode == "Direct reconstruction + pressure-constrained station peaks" and pressure_ready:
            station_table, peak_file = pressure_station_peaks(result, era5_path, track_path)
            summary = pd.concat([
                summary,
                pd.DataFrame({
                    "Product": ["Pressure-constrained station product"],
                    "Peak (m/s)": [float(station_table["pressure_constrained_peak_m_s"].max())],
                    "Increment over paired ERA5 (m/s)": [float(
                        station_table["pressure_constrained_peak_m_s"].max() -
                        station_table["ERA5_peak_m_s"].max()
                    )],
                }),
            ], ignore_index=True)
        elif mode == "Direct reconstruction + pressure-constrained station peaks":
            # Do not add an empty row: an event-peak calibration is undefined for
            # a short preview window, so the table should remain unambiguous.
            pass
        actual_device = load_runtime(requested_device)[1]
        backend_label = "ZeroGPU accelerated" if actual_device == "cuda" else "CPU quota-free"
        status = f"Completed {len(result['time'])} timestamp(s) using {backend_label}. "
        if mode == "Direct reconstruction + pressure-constrained station peaks":
            status += (
                "Pressure-constrained station peaks were generated for the supported CMA network."
                if pressure_ready else
                "Station-event peak correction requires the 24-hour animation option; no invalid "
                "single-time peak value was reported."
            )
        else:
            status += "Direct reconstruction uses the archived three-grid blended inference chain."
        return figure, peak_series, summary.round(3), str(direct_file), (
            str(peak_file) if peak_file else None
        ), status
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def infer_cpu(era5_file, track_file, duration, selected_timestamp, mode):
    return infer_impl(
        era5_file, track_file, duration, selected_timestamp, mode, requested_device="cpu"
    )


@gpu_task
def infer_gpu(era5_file, track_file, duration, selected_timestamp, mode):
    return infer_impl(
        era5_file, track_file, duration, selected_timestamp, mode, requested_device="cuda"
    )


CSS = """
.gradio-container {max-width: 1760px !important; margin: 0 auto; padding: 18px 22px 28px !important; color: #263744; background: #fff;}
#topbar {border-bottom: 1px solid #9CB2C3; padding: 0 2px 14px; margin-bottom: 16px;}
#topbar h1 {font-family: Georgia, 'Times New Roman', serif; font-size: 31px; color: #123D68; margin: 0 0 2px;}
#topbar p {font-size: 15px; color: #62798C; margin: 0;}
.section-title h3 {font-family: Georgia, 'Times New Roman', serif; font-size: 20px; color: #123D68; margin: 0 0 8px;}
#input-rail, #result-rail {border: 0; padding: 0 10px 0 2px;}
#map-stage {border-left: 1px solid #D4E0E8; border-right: 1px solid #D4E0E8; padding: 0 16px;}
.primary-btn {background: #0C4B83 !important; border-color: #0C4B83 !important; color: white !important; min-height: 48px; font-weight: 700 !important;}
.cpu-btn {background: #2F7655 !important; border-color: #2F7655 !important; color: white !important; min-height: 48px; font-weight: 700 !important;}
.download-row button, .download-row .wrap {border-color: #0C4B83 !important; color: #0C4B83 !important;}
.compact-note {font-size: 12px; color: #637789; line-height: 1.45;}
#scope-note {border-top: 1px solid #9CB2C3; margin-top: 16px; padding-top: 12px; color: #526879;}
table {font-size: 12px !important;}
footer {display: none !important;}
"""


initial_labels, initial_default_time = time_labels_and_default(EXAMPLE_ERA5)

with gr.Blocks(css=CSS, title="PhyTopo-Net Online Inference", theme=gr.themes.Base()) as demo:
    gr.HTML(
        "<div id='topbar'><h1>PhyTopo-Net Online Inference</h1>"
        "<p>Physics-informed topographic downscaling of coastal typhoon wind fields</p></div>"
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=3, min_width=300, elem_id="input-rail"):
            gr.Markdown("### Input", elem_classes="section-title")
            gr.Markdown("**1. Upload event data**")
            era5_input = gr.File(
                label="ERA5 U10/V10/MSL/T2/SST/BLH/FSR (.nc)",
                value=str(EXAMPLE_ERA5), file_types=[".nc"],
            )
            track_input = gr.File(
                label="Storm track and central pressure (.csv)",
                value=str(EXAMPLE_TRACK), file_types=[".csv"],
            )
            gr.Markdown("**2. Select product and time**")
            mode = gr.Radio(
                ["Direct reconstruction only",
                 "Direct reconstruction + pressure-constrained station peaks"],
                value="Direct reconstruction only",
                label="Inference mode",
            )
            duration = gr.Dropdown(
                ["Single timestamp", "6-hour animation", "24-hour animation"],
                value="Single timestamp", label="Output sequence",
            )
            selected_timestamp = gr.Dropdown(
                initial_labels, value=initial_default_time, label="Selected / centre time (UTC)",
                allow_custom_value=False,
            )
            with gr.Row():
                run_cpu = gr.Button("Run on CPU", elem_classes="cpu-btn")
                run_gpu = gr.Button("Run on ZeroGPU", variant="primary", elem_classes="primary-btn")
            gr.Markdown(
                "The six- and 24-hour choices process consecutive hourly fields centred on the "
                "selected time. Station-event peak correction is reported only for the 24-hour "
                "window; shorter windows never display a misleading calibrated peak. CPU mode "
                "does not consume ZeroGPU quota; ZeroGPU mode is faster but uses the visitor's "
                "daily allocation.",
                elem_classes="compact-note",
            )
        with gr.Column(scale=7, min_width=700, elem_id="map-stage"):
            gr.Markdown("### 10-m wind field", elem_classes="section-title")
            plot = gr.Plot(show_label=False)
            with gr.Row(elem_classes="download-row"):
                direct_download = gr.File(label="Download reconstructed NetCDF")
                peak_download = gr.File(label="Download station-peak CSV")
        with gr.Column(scale=3, min_width=330, elem_id="result-rail"):
            gr.Markdown("### Results", elem_classes="section-title")
            gr.Markdown("**Window peak evolution**")
            peak_plot = gr.Plot(show_label=False)
            gr.Markdown("**Peak comparison**")
            summary = gr.Dataframe(label="", interactive=False, wrap=True)
            gr.Markdown("**Status and provenance**")
            status = gr.Textbox(label="Status and provenance", interactive=False)
    gr.Markdown(
        "**Scope.** Direct reconstruction is defined on the bundled Pearl River Delta 3-km "
        "domain. The optional engineering product reports calibrated event peaks only for the "
        "supported CMA station network; it is not a universal arbitrary-grid bias correction. "
        "The map includes the locally supplied provincial and prefectural boundary layers.",
        elem_id="scope-note",
    )
    era5_input.change(timestamp_choices, inputs=[era5_input], outputs=[selected_timestamp])
    run_cpu.click(
        infer_cpu, inputs=[era5_input, track_input, duration, selected_timestamp, mode],
        outputs=[plot, peak_plot, summary, direct_download, peak_download, status],
    )
    run_gpu.click(
        infer_gpu, inputs=[era5_input, track_input, duration, selected_timestamp, mode],
        outputs=[plot, peak_plot, summary, direct_download, peak_download, status],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
