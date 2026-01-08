import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize_scalar

# --- PHYSICS ENGINE ---
G = 9.81


def simulate_trajectory(v0, angle_deg, mass_g, drag_k, wind_speed, dt=0.015):
    angle_rad = np.radians(angle_deg)
    m_kg = mass_g / 1000.0
    vx, vy = v0 * np.cos(angle_rad), v0 * np.sin(angle_rad)
    x, y = [0.0], [0.0]

    while y[-1] >= 0:
        v_rel_x = vx - wind_speed
        v_rel_y = vy
        v_mag = np.sqrt(v_rel_x ** 2 + v_rel_y ** 2)

        ax = -(drag_k * v_mag * v_rel_x) / m_kg
        ay = -G - (drag_k * v_mag * v_rel_y) / m_kg

        vx += ax * dt
        vy += ay * dt
        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)
        if len(x) > 5000: break
    return np.array(x), np.array(y)


def find_best_k(target_dist, v0, angle, mass):
    def objective(k):
        tx, ty = simulate_trajectory(v0, angle, mass, k, 0)
        return (tx[-1] - target_dist) ** 2

    res = minimize_scalar(objective, bounds=(0, 0.05), method='bounded')
    return res.x


# --- APP SETUP ---
st.set_page_config(page_title="Science Projectile Lab", layout="wide")
st.title("🚀 Science Projectile Lab")

if os.path.exists('experimental_data.csv'):
    df = pd.read_csv('experimental_data.csv')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[['Pressure', 'Angle', 'Mass']], df['Distance'])
    has_data = True
else:
    has_data = False
    st.error("Data file 'experimental_data.csv' not found.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Operation Mode")
    mode = st.radio("Select System Mode", ["Predict Distance", "Find Required Pressure", "Training Data View"],
                    label_visibility="collapsed")

    st.markdown("---")
    st.header("Flight Parameters")
    pressure = st.slider("Input Pressure", 10.0, 100.0, 50.0)
    angle = st.slider("Launch Angle (deg)", 5, 85, 45)
    ball_mass = st.number_input("Ball Weight (g)", value=10.0, step=1.0)

    st.markdown("---")
    st.header("Environment Settings")

    if 'wind_val' not in st.session_state: st.session_state.wind_val = 0.0
    if 'auto_k' not in st.session_state: st.session_state.auto_k = True
    if 'man_k' not in st.session_state: st.session_state.man_k = 0.002

    st.session_state.wind_val = st.slider("Wind Speed (m/s)", -10.0, 10.0, st.session_state.wind_val)
    st.session_state.auto_k = st.toggle("Auto-Calibrate Drag", value=st.session_state.auto_k)
    st.session_state.man_k = st.slider("Manual Drag (k)", 0.0, 0.015, st.session_state.man_k, step=0.00001,
                                       format="%.5f", disabled=st.session_state.auto_k)

    st.markdown("---")
    show_sensitivity = st.toggle("Show 5% Sensitivity Zone")

    st.markdown("---")
    st.header("Animation")
    animation_duration = st.slider("Animation Duration (sec)", 0.2, 5.0, 0.2, step=0.1,
                                   help="Time to draw each trajectory (0.2 = fast)")


# --- CHART UTILITIES ---
def draw_wind_compass(ax, wind_speed):
    if wind_speed == 0: return
    x_pos, y_pos = 0.5, 0.9
    arrow_len = 0.05 * abs(wind_speed)
    direction = 1 if wind_speed > 0 else -1

    ax.annotate('', xy=(x_pos + (arrow_len * direction), y_pos), xycoords='axes fraction',
                xytext=(x_pos, y_pos), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    label = "Tailwind" if wind_speed > 0 else "Headwind"
    ax.text(x_pos, y_pos + 0.02, f"{label}: {abs(wind_speed)}m/s",
            transform=ax.transAxes, ha='center', color='orange', fontsize=9, fontweight='bold')


def validate_trajectory_data(df):
    """
    Validate trajectory data for physics violations.
    Returns a boolean mask where True indicates valid entries.
    """
    # For projectile motion: max range = (v0^2 * sin(2*angle)) / g
    # At 45 degrees, sin(2*45) = 1, so max_range = v0^2 / g
    # Estimate v0 from pressure (assuming linear relationship)

    is_valid = pd.Series([True] * len(df), index=df.index)

    # Rule 1: Distance should be positive and reasonable (< 100m for most pressures)
    is_valid &= (df['Distance'] > 0) & (df['Distance'] < 150)

    # Rule 2: Angle should be between 0-90 degrees
    is_valid &= (df['Angle'] > 0) & (df['Angle'] <= 90)

    # Rule 3: Pressure should be positive
    is_valid &= (df['Pressure'] > 0)

    # Rule 4: Maximum distance roughly occurs near 45 degrees
    # Entries with very high distances at extreme angles (< 20° or > 70°) are suspect
    extreme_angles = (df['Angle'] < 20) | (df['Angle'] > 70)
    high_distance = df['Distance'] > df['Distance'].quantile(0.75)
    is_valid &= ~(extreme_angles & high_distance)

    # Rule 5: For same pressure/mass, distance should generally increase with angle up to 45°
    # Then decrease after 45° (quadratic-like behavior)
    # Check for dramatic outliers
    expected_max = df['Distance'].quantile(0.95)
    is_valid &= df['Distance'] <= expected_max * 1.2  # Allow 20% tolerance

    return is_valid



def plot_animated_trajectory(x_vacuum, y_vacuum, x_real, y_real, target_dist=None, wind_speed=0,
                            show_sensitivity=False, sensitivity_range=None, animation_duration=0.2):
    """
    Create an animated trajectory plot using Streamlit-compatible progressive drawing.

    The trajectory is drawn progressively by updating the plot multiple times, creating
    an animation effect without using matplotlib.animation (which doesn't work well in Streamlit).

    animation_duration: float - Total time in seconds to draw the complete trajectory
    """
    # Determine frame indices for ultra-fast rendering (cap at 20 frames)
    num_frames = min(len(x_real), 20)
    frame_indices = np.linspace(0, len(x_real) - 1, num_frames, dtype=int)

    # Create a placeholder for the animation
    plot_placeholder = st.empty()

    for frame in frame_indices:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # Set limits based on full trajectory
        ax.set_ylim(-1.0, max(y_vacuum.max(), y_real.max()) * 1.3)
        ax.set_xlim(0, max(x_vacuum[-1], x_real[-1], target_dist if target_dist else 0) * 1.1)

        # Plot sensitivity zone if provided
        if show_sensitivity and sensitivity_range:
            d_low, d_high = sensitivity_range
            ax.axvspan(d_low, d_high, color='yellow', alpha=0.2, label="±5% Var")

        # Plot vacuum trajectory (reference) - always shown
        ax.plot(x_vacuum, y_vacuum, color='gray', ls='--', alpha=0.3, label="Vacuum", linewidth=2)

        # Plot the trajectory up to current frame
        ax.plot(x_real[:frame + 1], y_real[:frame + 1], color='#007BFF', lw=2.5, label="Actual Path")

        # Show projectile marker at current position
        if frame > 0:
            ax.plot(x_real[frame], y_real[frame], marker='o', markersize=8, color='#007BFF',
                   markeredgecolor='black', markeredgewidth=1.5)

        # Target marker if provided
        if target_dist:
            ax.axvline(x=target_dist, color='red', linestyle=':', lw=1.5)
            ax.scatter([target_dist], [0], color='red', marker='X', s=100, zorder=10, clip_on=False, label="Target")

        draw_wind_compass(ax, wind_speed)
        ax.set_xlabel("Distance (m)", fontsize=10)
        ax.set_ylabel("Height (m)", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right')

        # Update the placeholder with the new figure (no delay for instant rendering)
        plot_placeholder.pyplot(fig)
        plt.close(fig)





# --- EXECUTION ---
if has_data and mode == "Predict Distance":
    ml_dist = model.predict(pd.DataFrame([[pressure, angle, ball_mass]], columns=['Pressure', 'Angle', 'Mass']))[0]
    v0_est = np.sqrt((ml_dist * G) / np.sin(np.radians(2 * angle)))
    k_to_use = find_best_k(ml_dist, v0_est * 1.2, angle,
                           ball_mass) if st.session_state.auto_k else st.session_state.man_k

    x_id, y_id = simulate_trajectory(v0_est * 1.2, angle, ball_mass, 0, 0)
    x_rl, y_rl = simulate_trajectory(v0_est * 1.2, angle, ball_mass, k_to_use, st.session_state.wind_val)

    # Display predictions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Prediction", f"{ml_dist:.2f}m")
    with col2:
        st.metric("Actual Distance", f"{x_rl[-1]:.2f}m")
    with col3:
        diff = abs(ml_dist - x_rl[-1])
        st.metric("Diff", f"{diff:.2f}m", delta=None)

    # Prepare sensitivity range if enabled
    sensitivity_range = None
    if show_sensitivity:
        low_p, high_p = pressure * 0.95, pressure * 1.05
        d_low = model.predict(pd.DataFrame([[low_p, angle, ball_mass]], columns=['Pressure', 'Angle', 'Mass']))[0]
        d_high = model.predict(pd.DataFrame([[high_p, angle, ball_mass]], columns=['Pressure', 'Angle', 'Mass']))[0]
        sensitivity_range = (d_low, d_high)

    plot_animated_trajectory(x_id, y_id, x_rl, y_rl, wind_speed=st.session_state.wind_val,
                            show_sensitivity=show_sensitivity, sensitivity_range=sensitivity_range,
                            animation_duration=animation_duration)

elif has_data and mode == "Find Required Pressure":
    st.subheader("🎯 Optimization: Find Pressure")
    target_dist = st.number_input("Target Distance (m)", value=25.0)

    p_test = np.linspace(10, 120, 100)
    best_p, min_err = 0, float('inf')

    for p in p_test:
        pred_d = model.predict(pd.DataFrame([[p, angle, ball_mass]], columns=['Pressure', 'Angle', 'Mass']))[0]
        v_test = np.sqrt((pred_d * G) / np.sin(np.radians(2 * angle)))
        test_k = find_best_k(pred_d, v_test * 1.2, angle,
                             ball_mass) if st.session_state.auto_k else st.session_state.man_k
        xw, _ = simulate_trajectory(v_test * 1.2, angle, ball_mass, test_k, st.session_state.wind_val)
        if abs(xw[-1] - target_dist) < min_err:
            min_err = abs(xw[-1] - target_dist)
            best_p = p

    st.success(f"Recommended Pressure: {best_p:.2f}")

    # RE-CALCULATE BOTH PATHS FOR DEMO
    base_ml_d = model.predict(pd.DataFrame([[best_p, angle, ball_mass]], columns=['Pressure', 'Angle', 'Mass']))[0]
    v0_final = np.sqrt((base_ml_d * G) / np.sin(np.radians(2 * angle)))
    k_final = find_best_k(base_ml_d, v0_final * 1.2, angle,
                          ball_mass) if st.session_state.auto_k else st.session_state.man_k

    # Reference path (no wind) and simulated path (with wind)
    xr, yr = simulate_trajectory(v0_final * 1.2, angle, ball_mass, k_final, 0)
    xf, yf = simulate_trajectory(v0_final * 1.2, angle, ball_mass, k_final, st.session_state.wind_val)

    # Display predictions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Distance", f"{target_dist:.2f}m")
    with col2:
        st.metric("AI Prediction", f"{base_ml_d:.2f}m")
    with col3:
        st.metric("Actual Distance", f"{xf[-1]:.2f}m")

    plot_animated_trajectory(xr, yr, xf, yf, target_dist=target_dist,
                            wind_speed=st.session_state.wind_val, animation_duration=animation_duration)

elif mode == "Training Data View":
    st.subheader("📋 Experimental Database")

    # Validate data and remove anomalies
    valid_mask = validate_trajectory_data(df)
    df = df[valid_mask].reset_index(drop=True)
    valid_mask = pd.Series([True] * len(df), index=df.index)

    st.info(f"Displaying {len(df)} valid test flights (anomalies removed)")

    # Create visualization tabs
    tab1, tab2, tab3 = st.tabs(["Pressure vs Distance", "Analysis", "Data Table"])

    with tab1:
        if len(df) > 0:
            # Create discrete angle bins (15° buckets)
            df_plot = df.copy()
            df_plot['Angle Bin'] = pd.cut(df_plot['Angle'], bins=[0, 15, 30, 45, 60, 75, 90],
                                          labels=['0-15°', '15-30°', '30-45°', '45-60°', '60-75°', '75-90°'],
                                          include_lowest=True)

            # Create Plotly scatter with discrete colors
            fig_scatter = px.scatter(
                df_plot,
                x='Pressure',
                y='Distance',
                color='Angle Bin',
                hover_data={'Pressure': ':.1f', 'Distance': ':.2f', 'Angle': ':.1f°', 'Mass': ':.2f g', 'Angle Bin': True},
                title='Projectile Distance vs Pressure',
                labels={'Pressure': 'Pressure', 'Distance': 'Distance (m)', 'Angle Bin': 'Launch Angle Range'},
                color_discrete_sequence=px.colors.qualitative.Set2,
                size_max=100
            )

            fig_scatter.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='black')))
            fig_scatter.update_layout(
                height=600,
                hovermode='closest',
                template='plotly_white',
                font=dict(size=11),
                showlegend=True,
                legend=dict(title='Launch Angle Range', x=1.02, y=1)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("No valid data to display")

    with tab2:
        # Statistical analysis
        st.write("**Distance Statistics by Angle Range:**")
        angle_ranges = [(0, 30), (30, 45), (45, 60), (60, 90)]
        stats_data = []

        for low, high in angle_ranges:
            mask = (df['Angle'] >= low) & (df['Angle'] < high)
            if mask.sum() > 0:
                subset = df[mask]['Distance']
                stats_data.append({
                    'Angle Range': f'{low}°-{high}°',
                    'Count': mask.sum(),
                    'Mean Distance': f"{subset.mean():.2f}m",
                    'Max Distance': f"{subset.max():.2f}m",
                    'Min Distance': f"{subset.min():.2f}m",
                    'Std Dev': f"{subset.std():.2f}m"
                })

        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    with tab3:
        # Filter option
        col1, col2 = st.columns(2)
        with col1:
            search_q = st.text_input("Filter by Pressure or Angle")

        # Apply filters
        display_df = df.copy()

        if search_q:
            display_df = display_df[
                display_df.apply(lambda row: search_q in str(row['Pressure']) or search_q in str(row['Angle']), axis=1)]

        st.dataframe(display_df, use_container_width=True)
