"""
FEA Destroyer — Structural Analysis Surrogate
Interactive web frontend for the trained GNN model.

Run:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FEA Destroyer",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from inference import (
    MATERIALS,
    make_mesh,
    build_data,
    load_model,
    run_inference,
)

# ── Model (cached across reruns) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


# ── Helpers ─────────────────────────────────────────────────────────────────
def correlation_badge(r):
    if r is None:
        return "⚪ Not evaluated"
    if r >= 0.99:
        return f"🟢 R = {r:.3f} — Production surrogate"
    if r >= 0.95:
        return f"🟢 R = {r:.3f} — Deployment ready"
    if r >= 0.90:
        return f"🟡 R = {r:.3f} — Screening only"
    if r >= 0.70:
        return f"🟠 R = {r:.3f} — Partial training — results are indicative"
    return f"🔴 R = {r:.3f} — Model not yet deployable"


def status_badge(min_sf, yield_mpa, max_stress_mpa):
    if max_stress_mpa >= yield_mpa:
        return "🔴 YIELDED", "error"
    if min_sf < 1.5:
        return "🟠 LOW SAFETY FACTOR", "warning"
    if min_sf < 2.5:
        return "🟡 ACCEPTABLE", "warning"
    return "🟢 SAFE", "success"


def _beam_edges(elements):
    """Extract unique edges from tetrahedral connectivity for wireframe rendering."""
    pair_idx = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    seen = set()
    edges = []
    for tet in elements:
        for i, j in pair_idx:
            key = (min(int(tet[i]), int(tet[j])), max(int(tet[i]), int(tet[j])))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return np.array(edges, dtype=np.int32)  # [E, 2]


def make_figure(nodes, elements, values, colorbar_label, warp_scale, disp_mm, fixed_mask):
    """
    Build a Plotly 3D FEA-style figure showing the deformed mesh coloured
    by the chosen scalar field (stress or displacement).
    """
    deformed = nodes + disp_mm * warp_scale / 1000.0   # back to metres

    edges = _beam_edges(elements)

    # Ghost wireframe — original shape
    orig_x, orig_y, orig_z = [], [], []
    for src, dst in edges:
        orig_x += [nodes[src, 0], nodes[dst, 0], None]
        orig_y += [nodes[src, 1], nodes[dst, 1], None]
        orig_z += [nodes[src, 2], nodes[dst, 2], None]

    # Deformed wireframe
    def_x, def_y, def_z = [], [], []
    for src, dst in edges:
        def_x += [deformed[src, 0], deformed[dst, 0], None]
        def_y += [deformed[src, 1], deformed[dst, 1], None]
        def_z += [deformed[src, 2], deformed[dst, 2], None]

    fixed = fixed_mask.reshape(-1) > 0.5

    fig = go.Figure()

    # Original geometry — subtle ghost
    fig.add_trace(go.Scatter3d(
        x=orig_x, y=orig_y, z=orig_z,
        mode="lines",
        line=dict(color="rgba(180,180,180,0.2)", width=1),
        name="Original shape",
        hoverinfo="none",
        showlegend=True,
    ))

    # Deformed wireframe
    fig.add_trace(go.Scatter3d(
        x=def_x, y=def_y, z=def_z,
        mode="lines",
        line=dict(color="rgba(100,100,100,0.4)", width=1),
        name="Deformed mesh",
        hoverinfo="none",
        showlegend=True,
    ))

    # Fixed nodes (clamp)
    if fixed.any():
        fig.add_trace(go.Scatter3d(
            x=nodes[fixed, 0],
            y=nodes[fixed, 1],
            z=nodes[fixed, 2],
            mode="markers",
            marker=dict(size=5, color="gray", symbol="square"),
            name="Fixed (clamped)",
        ))

    # Free nodes coloured by scalar field
    free = ~fixed
    hover_text = [f"{colorbar_label}: {v:.2f}" for v in values[free]]
    fig.add_trace(go.Scatter3d(
        x=deformed[free, 0],
        y=deformed[free, 1],
        z=deformed[free, 2],
        mode="markers",
        marker=dict(
            size=4,
            color=values[free],
            colorscale="Jet",
            colorbar=dict(
                title=dict(text=colorbar_label, side="right"),
                thickness=15,
            ),
            showscale=True,
            cmin=float(values[free].min()),
            cmax=float(values[free].max()),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name=colorbar_label,
    ))

    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X (m)", showbackground=False),
            yaxis=dict(title="Y (m)", showbackground=False),
            zaxis=dict(title="Z (m)  →  beam axis", showbackground=False),
            bgcolor="rgba(15,15,20,1)",
        ),
        paper_bgcolor="rgba(15,15,20,1)",
        plot_bgcolor="rgba(15,15,20,1)",
        font=dict(color="white"),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(40,40,40,0.7)",
            bordercolor="rgba(255,255,255,0.2)",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=560,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏗️ FEA Destroyer")
st.caption("Physics-informed GNN surrogate for cantilever beam structural analysis")

model, model_info = get_model()

# Model status banner
with st.container():
    if model is None:
        st.error(
            f"**{model_info.get('error', 'Model not loaded')}**  \n"
            "Run `python run_fresh.py` to generate data and train the model.",
            icon="⚠️",
        )
    else:
        r = model_info.get("correlation")
        badge = correlation_badge(r)
        epoch = model_info.get("epoch", "?")
        mae   = model_info.get("mae_mm")
        mae_str = f"  ·  MAE {mae:.3f} mm" if mae is not None else ""
        st.info(f"**Model status:** {badge}  ·  Epoch {epoch}{mae_str}", icon="ℹ️")

st.divider()

# ── Sidebar inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Beam Geometry")
    length = st.slider("Length (m)",  min_value=0.30, max_value=2.00, value=1.00, step=0.05)
    width  = st.slider("Width (m)",   min_value=0.03, max_value=0.30, value=0.08, step=0.01)
    height = st.slider("Height (m)",  min_value=0.03, max_value=0.30, value=0.10, step=0.01)

    st.markdown("---")
    st.header("Material")
    mat_name = st.selectbox("Material", list(MATERIALS.keys()))
    mat      = MATERIALS[mat_name]
    E, nu    = mat["E"], mat["nu"]
    yield_mpa = mat["yield_stress"] / 1e6

    st.caption(
        f"E = {E/1e9:.1f} GPa  ·  ν = {nu}  ·  Yield = {yield_mpa:.0f} MPa  ·  "
        f"ρ = {mat['density']} kg/m³"
    )

    st.markdown("---")
    st.header("Applied Force")
    F_kn   = st.slider("Magnitude (kN)", min_value=1.0,  max_value=200.0, value=10.0,  step=1.0)
    angle_x = st.slider("Angle X (°)",   min_value=-45,  max_value=45,    value=0,     step=1)
    angle_y = st.slider("Angle Y (°)",   min_value=-45,  max_value=45,    value=0,     step=1)

    F_mag = F_kn * 1000.0
    ax = np.radians(angle_x)
    ay = np.radians(angle_y)
    force_vec = np.array([
        F_mag * np.sin(ay),
        F_mag * np.sin(ax) * np.cos(ay),
        0.0,
    ])
    st.caption(
        f"Fx = {force_vec[0]/1000:.2f} kN  ·  "
        f"Fy = {force_vec[1]/1000:.2f} kN  ·  "
        f"Fz = {force_vec[2]/1000:.2f} kN"
    )

    st.markdown("---")
    st.header("Visualisation")
    color_by   = st.selectbox("Colour by", ["Von Mises Stress (MPa)", "Displacement (mm)"])
    warp_scale = st.slider("Deformation scale ×", min_value=1, max_value=500, value=50)
    run_btn    = st.button("▶  Run Analysis", type="primary", use_container_width=True)

# ── Main panel ───────────────────────────────────────────────────────────────
col_fig, col_metrics = st.columns([3, 1])

if run_btn:
    if model is None:
        st.error("Train the model first before running analysis.")
        st.stop()

    with st.spinner("Building mesh and running surrogate..."):
        nodes, elements = make_mesh(length, width, height, n=200)
        data = build_data(nodes, elements, force_vec, E, nu)
        results = run_inference(model, data)

    fixed_mask = data.fixed_mask.numpy()

    with col_fig:
        if color_by.startswith("Von Mises"):
            values    = results["stress_mpa"]
            cb_label  = "Stress (MPa)"
        else:
            values    = results["disp_mag_mm"]
            cb_label  = "Displacement (mm)"

        fig = make_figure(
            nodes, elements, values, cb_label,
            warp_scale, results["disp_mm"], fixed_mask,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Deformation shown at **{warp_scale}× scale**.  "
            f"Gray squares = fixed (clamped) end.  "
            f"Jet colourscale: blue → low, red → high."
        )

    with col_metrics:
        st.subheader("Results")

        max_d  = results["max_disp_mm"]
        max_s  = results["max_stress_mpa"]
        min_sf = results["min_sf"]
        label, kind = status_badge(min_sf, yield_mpa, max_s)

        if kind == "success":
            st.success(label)
        elif kind == "warning":
            st.warning(label)
        else:
            st.error(label)

        st.metric("Max Displacement",  f"{max_d:.3f} mm")
        st.metric("Max Von Mises Stress", f"{max_s:.1f} MPa",
                  delta=f"Yield: {yield_mpa:.0f} MPa", delta_color="off")
        st.metric("Min Safety Factor",  f"{min_sf:.2f}",
                  delta="Target ≥ 1.5", delta_color="off")

        st.divider()
        st.subheader("Beam Properties")
        L = length
        b = width
        h = height
        I  = b * h**3 / 12
        sigma_theory = (F_mag * L * (h / 2)) / I / 1e6
        delta_theory = (F_mag * L**3) / (3 * E * I) * 1000

        st.caption(
            f"**Theoretical (beam theory)**  \n"
            f"σ_max ≈ {sigma_theory:.1f} MPa  \n"
            f"δ_max ≈ {delta_theory:.3f} mm  \n\n"
            f"*Solid rectangular section assumed for theory.*"
        )

        st.divider()
        st.subheader("Disclaimer")
        r = model_info.get("correlation")
        if r is not None and r < 0.90:
            st.warning(
                "Model R < 0.90. These results are **indicative only**. "
                "Do not use for design decisions. Train longer or with more data.",
                icon="⚠️",
            )
        else:
            st.caption(
                "Results are surrogate predictions from a trained GNN, "
                "not a certified FEA solver. Always validate critical "
                "designs with full FEA and appropriate safety factors."
            )

else:
    with col_fig:
        st.info(
            "Set your beam geometry, material, and loading in the sidebar, "
            "then click **▶ Run Analysis**.",
            icon="👈",
        )

        # Preview beam outline
        lv, wv, hv = 1.0, 0.08, 0.10
        corners = np.array([
            [-wv/2, -hv/2, 0], [ wv/2, -hv/2, 0],
            [ wv/2,  hv/2, 0], [-wv/2,  hv/2, 0],
            [-wv/2, -hv/2, lv],[ wv/2, -hv/2, lv],
            [ wv/2,  hv/2, lv],[-wv/2,  hv/2, lv],
        ])
        lines = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]
        px, py, pz = [], [], []
        for a, b_ in lines:
            px += [corners[a,0], corners[b_,0], None]
            py += [corners[a,1], corners[b_,1], None]
            pz += [corners[a,2], corners[b_,2], None]

        preview = go.Figure(go.Scatter3d(
            x=px, y=py, z=pz, mode="lines",
            line=dict(color="steelblue", width=3),
        ))
        preview.update_layout(
            scene=dict(
                aspectmode="data",
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z (beam axis)",
                bgcolor="rgba(15,15,20,1)",
            ),
            paper_bgcolor="rgba(15,15,20,1)",
            font=dict(color="white"),
            margin=dict(l=0,r=0,t=10,b=0),
            height=400,
        )
        st.plotly_chart(preview, use_container_width=True)
        st.caption("Preview: default 1 m × 80 mm × 100 mm cantilever beam.")
