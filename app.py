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
col_title, col_help = st.columns([3, 1])
with col_title:
    st.title("Data Analyzer Pro")
if col_help.button("ℹ️ טיפים לשימוש בגרפים", use_container_width=True):
    st.info("""
    **הוראות שימוש בגרפים באפליקציה:**
    - 🔍 **מידע מפורט (Hover):** העבר את העכבר על הגרף כדי לראות את ערכי ה-X וה-Y של כל נקודה.
    - 👁️ **העלמת עמודות (Hide):** לחץ על שם השורה בצד ימין (Legend) כדי להעלים או לחשוף אותה בגרף. לחיצה כפולה תבודד רק אותה.
    - 🖱️ **זום ותזוזה:** גלול עם גלגלת העכבר כדי לעשות זום פנימה והחוצה. לחץ וגרור כדי להזיז את המיקוד.
    - 🎯 **איפוס ומירכוז:** עשית זום עמוק מדי? לחיצה כפולה (Double-Click) בכל מקום בגרף תמרכז אותו מחדש.
    """)

# Helper function to detect trigger pattern (all 0s, exactly one 1)
def is_trigger_column(series):
    if not pd.api.types.is_numeric_dtype(series):
        return False
    # Check if unique values are only 0 and 1
    uniques = series.dropna().unique()
    if set(uniques) <= {0, 1} or set(uniques) <= {0.0, 1.0}:
        # Check if 1 appears exactly once
        if (series == 1).sum() == 1:
            return True
    return False

# --- 1. Upload Data ---
st.header("1. Upload Data")
uploaded_files = st.file_uploader("Upload CSV / Excel Files (Multiple allowed)", type=['csv', 'xlsx', 'xls'], accept_multiple_files=True)

if uploaded_files:
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
        auto_trigger = st.checkbox("Auto-Detect Triggers (0-1-0)", value=True, help="Will automatically find columns with a single '1' and remaining '0's and annotate them")
    
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

    # --- 4. Interactive Trimming (Cut) ---
    st.header("4. Interactive Trimming (Cut Graph)")
    st.markdown("Use this to isolate a specific X-range for plotting and integral computation. **Note:** Only applies if X is numeric.")
    
    col_trim1, col_trim2, col_trim3, col_trim4 = st.columns(4)
    with col_trim1:
        trim_start = st.number_input("Start X (Trim)", value=0.0, step=1.0, key="trim_s")
    with col_trim2:
        trim_end = st.number_input("End X (Trim)", value=0.0, step=1.0, key="trim_e")
        
    # State for trim active
    if 'trim_active' not in st.session_state:
        st.session_state.trim_active = False

    with col_trim3:
        st.write("")
        st.write("")
        if st.button("Apply Trim (Cut)"):
            st.session_state.trim_active = True
    with col_trim4:
        st.write("")
        st.write("")
        if st.button("Reset Trim"):
            st.session_state.trim_active = False
            
    if st.session_state.trim_active:
        st.warning(f"✂️ Graph is currently trimmed between X = {trim_start} and X = {trim_end}. Integrals will only calculate for this region.")

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
        
        # Apply X offset
        if pd.api.types.is_numeric_dtype(df[x_col]):
            x_data = x_raw + config['x_offset']
            x_is_num = True
        else:
            x_data = np.arange(len(x_raw)) + config['x_offset']
            x_is_num = False
            
        # Optional: Auto-Detect Triggers in this file
        trigger_annotations = []
        if auto_trigger:
            for c in df.columns:
                if is_trigger_column(df[c]):
                    trigger_idx = df.index[df[c] == 1].tolist()[0]
                    trigger_x = x_data[trigger_idx]
                    trigger_y = 1 + config['y_offset'] # Approximate location
                    trigger_annotations.append({
                        'x': trigger_x,
                        'name': c
                    })

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
            
            # TRIMMING LOGIC
            plot_x_final = x_data
            plot_y_final = plot_y
            
            if st.session_state.trim_active and x_is_num:
                # Filter points where X is within range
                mask = (x_data >= trim_start) & (x_data <= trim_end)
                plot_x_final = x_data[mask]
                plot_y_final = plot_y[mask]
                
            # If after trimming there's no data, skip
            if len(plot_x_final) == 0:
                continue

            # Plot series
            fig.add_trace(go.Scatter(
                x=plot_x_final, 
                y=plot_y_final, 
                mode='lines', 
                name=trace_name,
                line=dict(color=colors[color_idx % len(colors)], width=2)
            ))
            
            # Average
            if show_average:
                avg_val = np.nanmean(plot_y_final)
                fig.add_hline(y=avg_val, line_dash="dash", line_color=colors[color_idx % len(colors)], 
                              annotation_text=f"Avg ({y_col}): {avg_val:.2f}")
            
            # Integral
            if calculate_integral:
                try:
                    if x_is_num:
                        sort_idx = np.argsort(plot_x_final)
                        area = simpson(y=plot_y_final[sort_idx], x=plot_x_final[sort_idx])
                        integral_results.append(f"**{trace_name}**: {area:.4f}")
                    else:
                        area = simpson(y=plot_y_final, dx=1.0)
                        integral_results.append(f"**{trace_name}**: {area:.4f} (dx=1)")
                except Exception as e:
                    integral_results.append(f"**{trace_name}**: Error computing area")
            
            color_idx += 1
            
        # Process trigger annotations onto the graph
        for trig in trigger_annotations:
            # Only show trigger if it falls within the trimmed region (if active)
            if st.session_state.trim_active and (trig['x'] < trim_start or trig['x'] > trim_end):
                continue
                
            fig.add_vline(x=trig['x'], line_dash="dot", line_color="red", 
                        annotation_text=f"Trigger: {trig['name']}<br>(X={trig['x']:.2f})", 
                        annotation_position="top right",
                        annotation_font_color="red")
            fig.add_trace(go.Scatter(
               x=[trig['x']],
               y=[0],
               mode='markers',
               marker=dict(color='red', size=12, symbol='star'),
               showlegend=False,
               hoverinfo="none"
            ))

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
        
        html_bytes = fig.to_html(include_plotlyjs="cdn", full_html=True)
        st.download_button(
            label="Download Graph (Interactive HTML)",
            data=html_bytes,
            file_name="interactive_graph.html",
            mime="text/html"
        )
