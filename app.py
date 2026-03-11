import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.integrate import simpson
import time

st.set_page_config(page_title="Data Analyzer Pro", layout="wide")

# --- Startup Animation ---
if 'startup_shown' not in st.session_state:
    st.session_state.startup_shown = False

if not st.session_state.startup_shown:
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            """
            <style>
            .fade-in-text {
                animation: fadeIn 2s;
                text-align: center;
                margin-top: 20vh;
                font-size: 3rem;
                font-weight: bold;
                color: #4CAF50;
            }
            @keyframes fadeIn {
                0% {opacity: 0;}
                100% {opacity: 1;}
            }
            </style>
            <div class="fade-in-text">✨ הפוך את המידע שלך לגרף חלומי ✨</div>
            """,
            unsafe_allow_html=True
        )
    time.sleep(3)  # Show animation for 3 seconds
    placeholder.empty() # Remove animation
    st.session_state.startup_shown = True

# Main App GUI
st.title("Data Analyzer Pro")

# --- 1. Upload Data ---
st.header("1. Upload Data")
uploaded_files = st.file_uploader("Upload CSV / Excel Files (Multiple allowed)", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)

if uploaded_files:
    # Read files into a dictionary of DataFrames
    datasets = {}
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                datasets[file.name] = pd.read_csv(file)
            else:
                datasets[file.name] = pd.read_excel(file)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    # --- 2. Configure Overlaps & Visualization ---
    st.header("2. Configure Visualization & Overlaps")
    
    # Global settings
    col_glb1, col_glb2, col_glb3 = st.columns(3)
    with col_glb1:
        graph_title = st.text_input("Graph Title:", "Data Visualization")
    with col_glb2:
        global_x_label = st.text_input("X-Axis Label:", "X")
    with col_glb3:
        global_y_label = st.text_input("Y-Axis Label:", "Values")

    st.write("### Configure Individual Files")
    
    file_configs = {}
    tabs = st.tabs(list(datasets.keys()))
    
    # Per-file configuration
    for i, (file_name, df) in enumerate(datasets.items()):
        with tabs[i]:
            col1, col2 = st.columns(2)
            all_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            with col1:
                x_axis = st.selectbox(f"X-Axis for {file_name}:", all_cols, key=f"x_{file_name}")
                x_offset = st.number_input(f"X Offset (Shift right/left):", value=0.0, step=1.0, format="%.4f", key=f"x_off_{file_name}")
            
            with col2:
                y_axes = st.multiselect(f"Y-Axes for {file_name} (Select multiple):", num_cols, default=num_cols[0] if num_cols else None, key=f"y_{file_name}")
                y_offset = st.number_input(f"Y Offset (Shift up/down):", value=0.0, step=1.0, format="%.4f", key=f"y_off_{file_name}")
            
            file_configs[file_name] = {
                'x_axis': x_axis,
                'y_axes': y_axes,
                'x_offset': x_offset,
                'y_offset': y_offset,
                'df': df
            }

    # --- 3. Analysis Tools ---
    st.header("3. Global Analysis Tools")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        show_average = st.checkbox("Show Average Line(s)")
    
    with col_t2:
        apply_smoothing = st.checkbox("Apply Smoothing")
        if apply_smoothing:
            smooth_type = st.selectbox("Smoothing Method", ["Rolling Average", "Savitzky-Golay"])
            window_size = st.slider("Window Size", min_value=3, max_value=51, step=2, value=5)
            poly_order = 3
            if smooth_type == "Savitzky-Golay":
                poly_order = st.slider("Polynomial Order (SG Only)", min_value=1, max_value=5, value=3)
    
    with col_t3:
        calculate_integral = st.checkbox("Calculate Integral (Area)")
    
    st.divider()

    # --- Graphing ---
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_idx = 0
    integral_results = []
    
    has_valid_data = False

    for file_name, config in file_configs.items():
        if not config['x_axis'] or not config['y_axes']:
            continue
            
        df = config['df']
        x_col = config['x_axis']
        x_raw = df[x_col].values
        
        # Apply X offset. (Offsets are easier to apply to numeric data).
        if pd.api.types.is_numeric_dtype(df[x_col]):
            x_data = x_raw + config['x_offset']
            x_is_num = True
        else:
            # If categorical, offset by index (not ideal, but workable for display)
            x_data = np.arange(len(x_raw)) + config['x_offset']
            x_is_num = False
            
        for y_col in config['y_axes']:
            has_valid_data = True
            y_data = df[y_col].values
            
            # Apply Y offset
            y_data_shifted = y_data + config['y_offset']
            plot_y = y_data_shifted.copy()
            name_suffix = ""
            
            # Smoothing
            if apply_smoothing:
                w = window_size
                if smooth_type == "Rolling Average":
                    plot_y = pd.Series(y_data_shifted).rolling(window=w, min_periods=1, center=True).mean().values
                    name_suffix = " (Rolling)"
                elif smooth_type == "Savitzky-Golay":
                    p = poly_order
                    if len(y_data_shifted) >= w and w > p:
                        if w % 2 == 0: w += 1
                        plot_y = savgol_filter(y_data_shifted, window_length=w, polyorder=p)
                        name_suffix = " (SG)"
            
            trace_name = f"{file_name} - {y_col}{name_suffix}"
            
            # Plot series
            fig.add_trace(go.Scatter(
                x=x_raw if not x_is_num else x_data, 
                y=plot_y, 
                mode='lines', 
                name=trace_name,
                line=dict(color=colors[color_idx % len(colors)], width=2)
            ))
            
            # Average
            if show_average:
                avg_val = np.nanmean(y_data_shifted)
                fig.add_hline(y=avg_val, line_dash="dash", line_color=colors[color_idx % len(colors)], 
                              annotation_text=f"Avg ({y_col}): {avg_val:.2f}")
            
            # Integral
            if calculate_integral:
                try:
                    if x_is_num:
                        sort_idx = np.argsort(x_data)
                        area = simpson(y=plot_y[sort_idx], x=x_data[sort_idx])
                        integral_results.append(f"**{trace_name}**: {area:.4f}")
                    else:
                        area = simpson(y=plot_y, dx=1.0)
                        integral_results.append(f"**{trace_name}**: {area:.4f} (dx=1)")
                except Exception as e:
                    integral_results.append(f"**{trace_name}**: Error computing area")
            
            color_idx += 1

    if has_valid_data:
        # Layout updates
        fig.update_layout(
            title=graph_title,
            xaxis_title=global_x_label,
            yaxis_title=global_y_label,
            legend_title="Series",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        if calculate_integral and integral_results:
            st.subheader("Integral Results")
            for res in integral_results:
                st.write(res)
                
        # --- Export Data Section ---
        st.divider()
        st.subheader("Export Center")
        st.write("*(Tip: To download the graph as an image, use the camera icon in the top right corner of the graph)*")
        
        # Export interactive HTML graph
        html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            label="Download Graph (Interactive HTML)",
            data=html_bytes,
            file_name="interactive_graph.html",
            mime="text/html"
        )
