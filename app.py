import streamlit as st
import pandas as pd
import numpy as np
import pyqtgraph as pg  # Used in old app, we'll use plotly or st.line_chart instead
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.integrate import simpson

st.set_page_config(page_title="Data Analyzer Pro", layout="wide")

st.title("Data Analyzer Pro")

# --- 1. Upload Data ---
st.header("1. Upload Data")
uploaded_file = st.file_uploader("Upload CSV / Excel File", type=['csv', 'xlsx', 'xls'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Loaded: {uploaded_file.name} | Rows: {len(df)}")
        
        # --- 2. Configure Visualization ---
        st.header("2. Configure Visualization")
        
        col1, col2 = st.columns(2)
        all_cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        with col1:
            x_axis = st.selectbox("X-Axis:", all_cols)
            graph_title = st.text_input("Graph Title:", "Data Visualization")
        
        with col2:
            y_axes = st.multiselect("Y-Axes (Select multiple):", num_cols, default=num_cols[0] if num_cols else None)
            x_label = st.text_input("X-Axis Label:", x_axis)
            y_label = st.text_input("Y-Axis Label:", "Values")
        
        # --- 3. Analysis Tools ---
        st.header("3. Analysis Tools")
        
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
        if x_axis and y_axes:
            
            x_raw = df[x_axis].values
            if pd.api.types.is_numeric_dtype(df[x_axis]):
                x_data = x_raw
                x_is_num = True
            else:
                x_data = np.arange(len(x_raw))
                x_is_num = False
                
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            
            integral_results = []
            
            for i, y_col in enumerate(y_axes):
                y_data = df[y_col].values
                plot_y = y_data.copy()
                name_suffix = ""
                
                # Smoothing
                if apply_smoothing:
                    w = window_size
                    if smooth_type == "Rolling Average":
                        plot_y = pd.Series(y_data).rolling(window=w, min_periods=1, center=True).mean().values
                        name_suffix = " (Rolling)"
                    elif smooth_type == "Savitzky-Golay":
                        p = poly_order
                        if len(y_data) >= w and w > p:
                            if w % 2 == 0: w += 1
                            plot_y = savgol_filter(y_data, window_length=w, polyorder=p)
                            name_suffix = " (SG)"
                
                # Plot series
                fig.add_trace(go.Scatter(
                    x=x_raw if not x_is_num else x_data, 
                    y=plot_y, 
                    mode='lines', 
                    name=f"{y_col}{name_suffix}",
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
                
                # Average
                if show_average:
                    avg_val = np.nanmean(y_data)
                    fig.add_hline(y=avg_val, line_dash="dash", line_color=colors[i % len(colors)], 
                                  annotation_text=f"Avg: {avg_val:.2f}")
                
                # Integral
                if calculate_integral:
                    try:
                        if x_is_num:
                            sort_idx = np.argsort(x_data)
                            area = simpson(y=plot_y[sort_idx], x=x_data[sort_idx])
                            integral_results.append(f"**{y_col}**: {area:.4f}")
                        else:
                            area = simpson(y=plot_y, dx=1.0)
                            integral_results.append(f"**{y_col}**: {area:.4f} (dx=1)")
                    except Exception as e:
                        integral_results.append(f"**{y_col}**: Error computing area")

            # Layout updates
            fig.update_layout(
                title=graph_title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                legend_title="Series"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            if calculate_integral and integral_results:
                st.subheader("Integral Results")
                for res in integral_results:
                    st.write(res)

    except Exception as e:
        st.error(f"Error: {e}")
