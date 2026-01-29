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
    """
    Simulate projectile trajectory with drag and wind.

    v0: (float) Initial velocity in m/s
    angle_deg: (float) Launch angle in degrees
    mass_g: (float) Projectile mass in grams
    drag_k: (float) Drag coefficient
    wind_speed: (float) Wind speed in m/s (positive = tailwind)
    dt: (float) Time step in seconds

    return: (tuple) Arrays of x and y positions
    """
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
    """
    Find optimal drag coefficient to match target distance.

    target_dist: (float) Target distance in meters
    v0: (float) Initial velocity in m/s
    angle: (float) Launch angle in degrees
    mass: (float) Projectile mass in grams

    return: (float) Optimal drag coefficient
    """
    def objective(k):
        tx, ty = simulate_trajectory(v0, angle, mass, k, 0)
        return (tx[-1] - target_dist) ** 2

    res = minimize_scalar(objective, bounds=(0, 0.05), method='bounded')
    return res.x


def calculate_spring_velocity(force_n, spring_k_npm):
    """
    Calculate initial velocity from spring launcher parameters.
    Uses energy conservation: (1/2)kx^2 = (1/2)mv^2

    force_n: (float) Launch force in Newtons
    spring_k_npm: (float) Spring constant in N/m

    return: (float) Initial velocity in m/s
    """
    if spring_k_npm <= 0:
        return 0
    # F = kx, so x = F/k (compression distance)
    compression = force_n / spring_k_npm
    # Spring energy: E = (1/2)kx^2
    # For typical projectile mass ~10g = 0.01kg
    # v = sqrt(k * x^2 / m)
    mass_kg = 0.01  # Standard mass for spring launcher
    velocity = np.sqrt((spring_k_npm * compression ** 2) / mass_kg)
    return velocity


# --- APP SETUP ---
st.set_page_config(page_title="Science Projectile Lab", layout="wide")

# Compact sidebar styling to reduce spacing
st.markdown(
    """
    <style>
        /* Reduce overall sidebar padding */
        [data-testid="stSidebar"] .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
        /* Tighten vertical spacing between elements in the sidebar */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stRadio, [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stToggle, [data-testid="stSidebar"] .stButton, [data-testid="stSidebar"] .stNumberInput,
        [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stTextInput, [data-testid="stSidebar"] .stMetric {
            margin-top: 0.25rem; margin-bottom: 0.25rem;
        }
        /* Make captions compact */
        [data-testid="stSidebar"] .stMarkdown p { margin: 0.2rem 0; }
        /* Reduce metric spacing if used in sidebar later */
        [data-testid="stSidebar"] .stMetric { padding: 0.25rem 0; }

        /* Apply compact styling to main content area */
        [data-testid="stAppViewContainer"] .block-container { padding-top: 0.75rem; padding-bottom: 0.75rem; }
        [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] .stMarkdown, [data-testid="stAppViewContainer"] .stRadio, [data-testid="stAppViewContainer"] .stSlider,
        [data-testid="stAppViewContainer"] .stToggle, [data-testid="stAppViewContainer"] .stButton, [data-testid="stAppViewContainer"] .stNumberInput,
        [data-testid="stAppViewContainer"] .stSelectbox, [data-testid="stAppViewContainer"] .stTextInput, [data-testid="stAppViewContainer"] .stMetric,
        [data-testid="stAppViewContainer"] .stTabs, [data-testid="stAppViewContainer"] .stDataFrame {
            margin-top: 0.35rem; margin-bottom: 0.35rem;
        }
        /* Compact markdown paragraphs in main area */
        [data-testid="stAppViewContainer"] .stMarkdown p { margin: 0.25rem 0; }
        /* Compact tabs header spacing */
        [data-testid="stAppViewContainer"] .stTabs [role="tablist"] { margin-bottom: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Add one empty line at the top of the main page content
st.write("")

# Load and prepare experimental data
if os.path.exists('experimental_data.csv'):
    df_raw = pd.read_csv('experimental_data.csv')

    # Separate the two test types
    df_pump = df_raw[df_raw['Test'].str.contains('Gas Pump', na=False)].copy() if len(df_raw) > 0 else pd.DataFrame()
    df_spring = df_raw[df_raw['Test'].str.contains('Spring', na=False)].copy() if len(df_raw) > 0 else pd.DataFrame()

    # Prepare pump data (Test 1) - uses angles
    if len(df_pump) > 0:
        df_pump['TestType'] = 'Pump'
        df_pump['Pressure'] = 50.0  # Fixed pressure for pump tests
        df_pump['Mass'] = 10.0  # Standard ball mass in grams
        df_pump = df_pump.rename(columns={'Distance_m': 'Distance', 'Angle_deg': 'Angle'})

    # Prepare spring data (Test 2) - uses force and spring constant
    if len(df_spring) > 0:
        df_spring['TestType'] = 'Spring'
        df_spring['Angle'] = 45.0  # Fixed angle for spring tests
        df_spring['Mass'] = 10.0  # Standard ball mass in grams
        df_spring = df_spring.rename(columns={'Distance_m': 'Distance',
                                              'LaunchForce_N': 'Force',
                                              'SpringConstant_Npm': 'SpringK'})
        # Calculate equivalent "pressure" from spring parameters for unified model
        df_spring['Pressure'] = df_spring['Force'] * df_spring['SpringK'] * 5

    # Combine datasets
    df = pd.concat([df_pump, df_spring], ignore_index=True)

    # Train ML model on combined data
    model_pump = None
    model_spring = None

    if len(df_pump) > 0:
        model_pump = RandomForestRegressor(n_estimators=100, random_state=42)
        model_pump.fit(df_pump[['Angle']], df_pump['Distance'])

    if len(df_spring) > 0:
        model_spring = RandomForestRegressor(n_estimators=100, random_state=42)
        model_spring.fit(df_spring[['Force', 'SpringK']], df_spring['Distance'])

    has_data = True
    has_pump_data = len(df_pump) > 0
    has_spring_data = len(df_spring) > 0
else:
    has_data = False
    has_pump_data = False
    has_spring_data = False
    st.error("Data file 'experimental_data.csv' not found.")

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚀 Science Projectile Lab")
    # Removed horizontal rule to save vertical space

    # Operation Mode (compact)
    st.caption("Operation Mode")
    mode = st.radio("Select System Mode", ["Predict Distance", "Find Required Input", "Training Data View"],
                    label_visibility="collapsed")

    # Test Type (compact)
    st.caption("Test Type")
    test_type = st.radio("Select Test Type", ["Gas Pump (Angle-based)", "Spring Launcher (Force-based)"],
                         disabled=not has_data)

    # Test-specific parameters
    if test_type == "Gas Pump (Angle-based)":
        st.caption("Gas Pump Parameters")
        pressure = 50.0  # Fixed for pump tests
        angle = st.slider("Launch Angle (deg)", 5, 85, 45)
        ball_mass = 10.0  # Fixed for pump tests
        st.info(f"Fixed Pressure: {pressure} | Mass: {ball_mass}g")

    else:  # Spring Launcher
        st.caption("Spring Launcher Parameters")
        # Hide launch force slider in "Find Required Input" mode
        if mode != "Find Required Input":
            launch_force = st.slider("Launch Force (N)", 1.0, 40.0, 4.0, step=0.5)
            spring_constant = st.slider("Spring Constant (N/m)", 10.0, 40.0, 20.0, step=1.0)
            # Calculate velocity from spring parameters
            spring_velocity = calculate_spring_velocity(launch_force, spring_constant)
            st.metric("Calculated Velocity", f"{spring_velocity:.2f} m/s")
        else:
            # In "Find Required Input" mode, only show spring constant
            spring_constant = st.slider("Spring Constant (N/m)", 10.0, 40.0, 20.0, step=1.0)

        angle = 45.0  # Fixed for spring tests
        ball_mass = 10.0  # Fixed for spring tests
        st.info(f"Fixed Angle: {angle}° | Mass: {ball_mass}g")

    # Environment Settings (compact)
    st.caption("Environment Settings")

    if 'wind_val' not in st.session_state: st.session_state.wind_val = 0.0
    if 'auto_k' not in st.session_state: st.session_state.auto_k = False  # default disabled
    if 'man_k' not in st.session_state: st.session_state.man_k = 0.0001
    if 'wind_changed' not in st.session_state: st.session_state.wind_changed = False

    # Wind speed slider with callback to force animation refresh
    new_wind = st.slider("Wind Speed (m/s)", -10.0, 10.0, st.session_state.wind_val, key="wind_slider")
    if new_wind != st.session_state.wind_val:
        st.session_state.wind_val = new_wind
        st.session_state.wind_changed = not st.session_state.wind_changed  # Toggle to force rerun
        st.rerun()  # Force animation to update with new wind speed

    st.session_state.auto_k = st.toggle("Auto-Calibrate Drag", value=st.session_state.auto_k)
    st.session_state.man_k = st.slider("Manual Drag (k)", 0.0, 0.015, st.session_state.man_k, step=0.00001,
                                       format="%.5f", disabled=st.session_state.auto_k)

    # Sensitivity toggle (compact)
    show_sensitivity = st.toggle("Show 5% Sensitivity Zone")

    # Animation settings (compact)
    st.caption("Animation")
    animation_duration = st.slider("Animation Duration (sec)", 0.2, 5.0, 0.2, step=0.1,
                                   help="Time to draw each trajectory (0.2 = fast)")


# --- CHART UTILITIES ---
def draw_wind_compass(ax, wind_speed):
    """
    Draw a wind direction indicator on the plot.

    ax: (matplotlib.axes.Axes) The axes to draw on
    wind_speed: (float) Wind speed in m/s
    """
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


def validate_pump_data(df_pump):
    """
    Validate gas pump trajectory data for physics violations.

    df_pump: (DataFrame) Gas pump test data with Angle and Distance columns

    return: (Series) Boolean mask where True indicates valid entries
    """
    is_valid = pd.Series([True] * len(df_pump), index=df_pump.index)

    # Rule 1: Distance should be positive and reasonable
    is_valid &= (df_pump['Distance'] > 0) & (df_pump['Distance'] < 15)

    # Rule 2: Angle should be between 5-85 degrees
    is_valid &= (df_pump['Angle'] >= 5) & (df_pump['Angle'] <= 85)

    # Rule 3: Remove only extreme outliers (more than 2 standard deviations from mean per angle)
    for angle in df_pump['Angle'].unique():
        angle_mask = df_pump['Angle'] == angle
        if angle_mask.sum() > 2:
            subset = df_pump[angle_mask]['Distance']
            mean_dist = subset.mean()
            std_dist = subset.std()
            # Keep data within 2.5 standard deviations
            is_valid &= ~((df_pump['Angle'] == angle) &
                         ((df_pump['Distance'] < mean_dist - 2.5 * std_dist) |
                          (df_pump['Distance'] > mean_dist + 2.5 * std_dist)))

    return is_valid


def validate_spring_data(df_spring):
    """
    Validate spring launcher trajectory data for physics violations.

    df_spring: (DataFrame) Spring launcher test data with Force, SpringK, and Distance columns

    return: (Series) Boolean mask where True indicates valid entries
    """
    is_valid = pd.Series([True] * len(df_spring), index=df_spring.index)

    # Rule 1: Distance should be positive and reasonable
    is_valid &= (df_spring['Distance'] > 0) & (df_spring['Distance'] < 2.0)

    # Rule 2: Force should be positive and within reasonable range
    is_valid &= (df_spring['Force'] > 0) & (df_spring['Force'] <= 50)

    # Rule 3: Spring constant should be positive and within reasonable range
    is_valid &= (df_spring['SpringK'] > 0) & (df_spring['SpringK'] <= 100)

    # That's it - accept all data that meets these basic physical constraints
    # No additional outlier removal for Spring data to keep all valid measurements

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

    # Create a placeholder for the animation with unique key based on wind speed to prevent caching
    # This ensures the animation updates when wind speed changes
    plot_placeholder = st.empty()

    # Calculate appropriate axis limits based on actual trajectory range
    max_x = max(x_vacuum[-1], x_real[-1], target_dist if target_dist else 0)
    max_y = max(y_vacuum.max(), y_real.max())

    # Add 20% padding to both axes for better visualization
    x_limit = max_x * 1.2
    y_limit = max_y * 1.3

    # Ensure minimum reasonable limits to prevent tiny plots
    # For small trajectories (Spring Launcher), use a minimum of 0.5m
    x_limit = max(x_limit, 0.5)
    y_limit = max(y_limit, 0.1)

    for frame in frame_indices:
        fig, ax = plt.subplots(figsize=(10, 4.5))

        # Set limits based on calculated range (not fixed 12m)
        ax.set_ylim(0, y_limit)
        ax.set_xlim(0, x_limit)

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
    # Determine which model and data to use
    if test_type == "Gas Pump (Angle-based)" and has_pump_data:
        ml_dist = model_pump.predict(pd.DataFrame([[angle]], columns=['Angle']))[0]
        v0_est = np.sqrt((ml_dist * G) / np.sin(np.radians(2 * angle)))
    elif test_type == "Spring Launcher (Force-based)" and has_spring_data:
        ml_dist = model_spring.predict(pd.DataFrame([[launch_force, spring_constant]],
                                                    columns=['Force', 'SpringK']))[0]
        # Use ML prediction to calculate proper velocity instead of raw spring velocity
        # This ensures trajectory matches the predicted distance
        v0_est = np.sqrt((ml_dist * G) / np.sin(np.radians(2 * angle)))
    else:
        st.error(f"No data available for {test_type}")
        st.stop()

    k_to_use = find_best_k(ml_dist, v0_est, angle,
                           ball_mass) if st.session_state.auto_k else st.session_state.man_k

    x_id, y_id = simulate_trajectory(v0_est, angle, ball_mass, 0, 0)
    x_rl, y_rl = simulate_trajectory(v0_est, angle, ball_mass, k_to_use, st.session_state.wind_val)

    # Calculate difference and format appropriately
    actual_distance = x_rl[-1]
    diff = abs(ml_dist - actual_distance)

    # Format difference - use cm if less than 0.01m, otherwise use m
    if diff < 0.01:
        diff_display = f"{diff * 100:.1f}cm"
    else:
        diff_display = f"{diff:.2f}m"

    # Display predictions with shorter labels for Spring Launcher
    if test_type == "Spring Launcher (Force-based)":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("AI 🤖", f"{ml_dist:.3f}m", help="Machine learning model prediction")
        with col2:
            st.metric("Actual 📍", f"{actual_distance:.3f}m", help="Physics simulation result")
        with col3:
            st.metric("Diff 📊", diff_display, help="Absolute difference between prediction and actual")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🤖 AI Prediction", f"{ml_dist:.2f}m", help="Machine learning model prediction")
        with col2:
            st.metric("📍 Actual Distance", f"{actual_distance:.2f}m", help="Physics simulation result")
        with col3:
            st.metric("📊 Difference", diff_display, help="Absolute difference between prediction and actual")

    # Prepare sensitivity range if enabled
    sensitivity_range = None
    if show_sensitivity:
        if test_type == "Gas Pump (Angle-based)":
            low_a, high_a = angle * 0.95, angle * 1.05
            d_low = model_pump.predict(pd.DataFrame([[low_a]], columns=['Angle']))[0]
            d_high = model_pump.predict(pd.DataFrame([[high_a]], columns=['Angle']))[0]
        else:  # Spring Launcher
            low_f, high_f = launch_force * 0.95, launch_force * 1.05
            d_low = model_spring.predict(pd.DataFrame([[low_f, spring_constant]],
                                                      columns=['Force', 'SpringK']))[0]
            d_high = model_spring.predict(pd.DataFrame([[high_f, spring_constant]],
                                                       columns=['Force', 'SpringK']))[0]
        sensitivity_range = (d_low, d_high)

    plot_animated_trajectory(x_id, y_id, x_rl, y_rl, wind_speed=st.session_state.wind_val,
                            show_sensitivity=show_sensitivity, sensitivity_range=sensitivity_range,
                            animation_duration=animation_duration)

elif has_data and mode == "Find Required Input":
    st.subheader("🎯 Optimization: Find Required Input")

    # Use fixed reasonable defaults
    if test_type == "Gas Pump (Angle-based)" and has_pump_data:
        default_target = 6.0
        min_target = 3.0
        max_target = 8.0
    elif test_type == "Spring Launcher (Force-based)" and has_spring_data:
        default_target = 0.2
        min_target = 0.05
        max_target = 0.6
    else:
        default_target = 5.0
        min_target = 0.1
        max_target = 50.0

    # Create narrower input using columns
    col_input, _ = st.columns([1, 2])
    with col_input:
        target_dist = st.number_input("Target Distance (m)", value=default_target, min_value=min_target, max_value=max_target)

    if test_type == "Gas Pump (Angle-based)" and has_pump_data:
        # Find best angle
        angle_test = np.linspace(15, 75, 100)
        best_angle, min_err = 0, float('inf')

        for a in angle_test:
            pred_d = model_pump.predict(pd.DataFrame([[a]], columns=['Angle']))[0]
            v_test = np.sqrt((pred_d * G) / np.sin(np.radians(2 * a)))
            test_k = find_best_k(pred_d, v_test, a,
                                 ball_mass) if st.session_state.auto_k else st.session_state.man_k
            xw, _ = simulate_trajectory(v_test, a, ball_mass, test_k, st.session_state.wind_val)
            if abs(xw[-1] - target_dist) < min_err:
                min_err = abs(xw[-1] - target_dist)
                best_angle = a

        st.success(f"Recommended Angle: {best_angle:.1f}°")

        # Calculate final trajectory without the 1.2x multiplier
        base_ml_d = model_pump.predict(pd.DataFrame([[best_angle]], columns=['Angle']))[0]
        v0_final = np.sqrt((base_ml_d * G) / np.sin(np.radians(2 * best_angle)))
        k_final = find_best_k(base_ml_d, v0_final, best_angle,
                              ball_mass) if st.session_state.auto_k else st.session_state.man_k

        # Reference path (no wind) and simulated path (with wind)
        xr, yr = simulate_trajectory(v0_final, best_angle, ball_mass, k_final, 0)
        xf, yf = simulate_trajectory(v0_final, best_angle, ball_mass, k_final, st.session_state.wind_val)

        # Display predictions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Distance", f"{target_dist:.2f}m")
        with col2:
            st.metric("AI Prediction", f"{base_ml_d:.2f}m")
        with col3:
            st.metric("Actual Distance", f"{xf[-1]:.2f}m")

    elif test_type == "Spring Launcher (Force-based)" and has_spring_data:
        # Find best force
        force_test = np.linspace(2, 40, 100)  # Updated to match new force range
        best_force, min_err = 0, float('inf')

        for f in force_test:
            pred_d = model_spring.predict(pd.DataFrame([[f, spring_constant]], columns=['Force', 'SpringK']))[0]
            # Use ML prediction to calculate velocity (same as Predict Distance mode)
            v_test = np.sqrt((pred_d * G) / np.sin(np.radians(2 * angle)))
            test_k = find_best_k(pred_d, v_test, angle, ball_mass) if st.session_state.auto_k else st.session_state.man_k
            xw, _ = simulate_trajectory(v_test, angle, ball_mass, test_k, st.session_state.wind_val)
            if abs(xw[-1] - target_dist) < min_err:
                min_err = abs(xw[-1] - target_dist)
                best_force = f

        st.success(f"Recommended Force: {best_force:.2f} N")

        # Calculate final trajectory using ML prediction-based velocity
        base_ml_d = model_spring.predict(pd.DataFrame([[best_force, spring_constant]],
                                                      columns=['Force', 'SpringK']))[0]
        v0_final = np.sqrt((base_ml_d * G) / np.sin(np.radians(2 * angle)))
        k_final = find_best_k(base_ml_d, v0_final, angle,
                              ball_mass) if st.session_state.auto_k else st.session_state.man_k

        # Reference path (no wind) and simulated path (with wind)
        xr, yr = simulate_trajectory(v0_final, angle, ball_mass, k_final, 0)
        xf, yf = simulate_trajectory(v0_final, angle, ball_mass, k_final, st.session_state.wind_val)

        # Display predictions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Distance", f"{target_dist:.2f}m")
        with col2:
            st.metric("AI Prediction", f"{base_ml_d:.2f}m")
        with col3:
            st.metric("Actual Distance", f"{xf[-1]:.2f}m")
    else:
        st.error(f"No data available for {test_type}")
        st.stop()

    plot_animated_trajectory(xr, yr, xf, yf, target_dist=target_dist,
                            wind_speed=st.session_state.wind_val, animation_duration=animation_duration)

elif mode == "Training Data View":
    # Show test type selector - radio only, no label text
    view_test_type = st.radio(" ",
                              ["Gas Pump Tests", "Spring Launcher Tests", "Both"],
                              horizontal=True)

    # Prepare data based on selection
    if view_test_type == "Gas Pump Tests" and has_pump_data:
        valid_mask = validate_pump_data(df_pump)
        df_display = df_pump[valid_mask].reset_index(drop=True)
        test_name = "Gas Pump (Angle-based)"
    elif view_test_type == "Spring Launcher Tests" and has_spring_data:
        valid_mask = validate_spring_data(df_spring)
        df_display = df_spring[valid_mask].reset_index(drop=True)
        test_name = "Spring Launcher (Force-based)"
    elif view_test_type == "Both":
        if has_pump_data:
            valid_pump = validate_pump_data(df_pump)
            df_pump_clean = df_pump[valid_pump].reset_index(drop=True)
        else:
            df_pump_clean = pd.DataFrame()

        if has_spring_data:
            valid_spring = validate_spring_data(df_spring)
            df_spring_clean = df_spring[valid_spring].reset_index(drop=True)
        else:
            df_spring_clean = pd.DataFrame()

        df_display = pd.concat([df_pump_clean, df_spring_clean], ignore_index=True)
        test_name = "All Tests"
    else:
        st.error(f"No data available for {view_test_type}")
        st.stop()


    # Create visualization tabs
    tab1, tab2, tab3 = st.tabs(["Primary Analysis", "Statistical Analysis", "Data Table"])

    with tab1:
        if len(df_display) > 0:
            if view_test_type == "Gas Pump Tests" or (view_test_type == "Both" and has_pump_data):
                # Create enhanced visualization for Gas Pump data
                df_pump_vis = df_display[df_display['TestType'] == 'Pump'].copy() if 'TestType' in df_display.columns else df_display.copy()

                if len(df_pump_vis) > 0:
                    # Calculate statistics by angle for trend line and confidence bands
                    angle_stats = df_pump_vis.groupby('Angle')['Distance'].agg([
                        'mean', 'std', 'count', 'min', 'max'
                    ]).reset_index()
                    angle_stats['se'] = angle_stats['std'] / np.sqrt(angle_stats['count'])
                    angle_stats['ci'] = 1.96 * angle_stats['se']  # 95% confidence interval

                    # Display summary statistics FIRST
                    st.write("**Summary Statistics**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        max_angle = angle_stats.loc[angle_stats['mean'].idxmax(), 'Angle']
                        st.metric("Optimal Angle", f"{max_angle:.0f}°")
                    with col2:
                        max_dist = angle_stats['mean'].max()
                        st.metric("Max Distance", f"{max_dist:.3f}m")
                    with col3:
                        total_trials = len(df_pump_vis)
                        st.metric("Total Trials", f"{total_trials}")
                    with col4:
                        num_angles = df_pump_vis['Angle'].nunique()
                        st.metric("Angles Tested", f"{num_angles}")

                    st.markdown("---")

                    # Create line plot with confidence bands
                    fig_pump = px.line(
                        angle_stats,
                        x='Angle',
                        y='mean',
                        hover_data={
                            'Angle': ':.0f°',
                            'mean': ':.3f m',
                            'std': ':.3f m',
                            'count': ':d',
                            'min': ':.3f m',
                            'max': ':.3f m'
                        },
                        labels={'Angle': 'Launch Angle (degrees)', 'mean': 'Average Distance (m)'},
                        title='Gas Pump: Projectile Range vs Launch Angle',
                        markers=True,
                        line_shape='spline'
                    )

                    # Add confidence band (upper and lower bounds)
                    fig_pump.add_scatter(
                        x=angle_stats['Angle'],
                        y=angle_stats['mean'] + angle_stats['ci'],
                        mode='lines',
                        name='95% Confidence Band',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                    fig_pump.add_scatter(
                        x=angle_stats['Angle'],
                        y=angle_stats['mean'] - angle_stats['ci'],
                        mode='lines',
                        name='95% Confidence Band',
                        line=dict(width=0),
                        fillcolor='rgba(0, 123, 255, 0.2)',
                        fill='tonexty',
                        showlegend=True,
                        hoverinfo='skip'
                    )

                    # Add individual data points as scatter
                    fig_pump.add_scatter(
                        x=df_pump_vis['Angle'],
                        y=df_pump_vis['Distance'],
                        mode='markers',
                        name='Individual Trials',
                        marker=dict(size=6, color='rgba(0, 123, 255, 0.4)',
                                   line=dict(width=1, color='darkblue')),
                        hovertemplate='<b>Angle:</b> %{x:.0f}°<br><b>Distance:</b> %{y:.3f}m<extra></extra>'
                    )

                    # Update layout for better visualization
                    fig_pump.update_layout(
                        height=550,
                        template='plotly_white',
                        font=dict(size=12),
                        hovermode='closest',
                        xaxis=dict(title='Launch Angle (degrees)', dtick=5),
                        yaxis=dict(title='Distance (m)'),
                        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
                    )
                    fig_pump.update_traces(
                        line=dict(color='#0078D4', width=3),
                        selector=dict(mode='lines')
                    )

                    st.plotly_chart(fig_pump, use_container_width=True)


            if view_test_type == "Spring Launcher Tests" or (view_test_type == "Both" and has_spring_data):
                # Create 3D visualization for spring data with force, spring constant, and distance
                df_spring_vis = df_display[df_display['TestType'] == 'Spring'].copy() if 'TestType' in df_display.columns else df_display.copy()

                if len(df_spring_vis) > 0:
                    # Use 3D scatter to show Force, SpringK, and Distance relationships
                    fig_spring = px.scatter_3d(
                        df_spring_vis,
                        x='Force',
                        y='SpringK',
                        z='Distance',
                        color='Distance',
                        hover_data={'Force': ':.1f N', 'SpringK': ':.1f N/m', 'Distance': ':.3f m'},
                        labels={'Force': 'Launch Force (N)', 'SpringK': 'Spring Constant (N/m)', 'Distance': 'Distance (m)'},
                        color_continuous_scale='Viridis',
                        size_max=8,
                        title='Spring Launcher: Force × Spring Constant Effect on Distance<br><sub>Fixed Launch Angle: 45°</sub>'
                    )
                    fig_spring.update_layout(
                        height=600,
                        template='plotly_white',
                        font=dict(size=11),
                        scene=dict(
                            xaxis_title='Launch Force (N)',
                            yaxis_title='Spring Constant (N/m)',
                            zaxis_title='Distance (m)'
                        )
                    )
                    st.plotly_chart(fig_spring, use_container_width=True)

                    # Display the fixed angle info clearly
                    st.info("📌 **Fixed Launch Angle: 45°** - This optimal angle from projectile physics theory is maintained constant while varying force and spring constant parameters.")
        else:
            st.warning("No valid data to display")

    with tab2:
        # Statistical analysis
        if view_test_type == "Gas Pump Tests" or (view_test_type == "Both" and has_pump_data):
            df_pump_stats = df_display[df_display['TestType'] == 'Pump'].copy() if 'TestType' in df_display.columns else df_display.copy()

            if len(df_pump_stats) > 0:
                st.write("**Gas Pump: Distance Statistics by Angle Range**")
                angle_ranges = [(5, 20), (20, 35), (35, 50), (50, 65), (65, 85)]
                stats_data = []

                for low, high in angle_ranges:
                    mask = (df_pump_stats['Angle'] >= low) & (df_pump_stats['Angle'] < high)
                    if mask.sum() > 0:
                        subset = df_pump_stats[mask]['Distance']
                        stats_data.append({
                            'Angle Range': f'{low}°-{high}°',
                            'Count': mask.sum(),
                            'Mean Distance': f"{subset.mean():.3f}m",
                            'Max Distance': f"{subset.max():.3f}m",
                            'Min Distance': f"{subset.min():.3f}m",
                            'Std Dev': f"{subset.std():.3f}m"
                        })

                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        if view_test_type == "Spring Launcher Tests" or (view_test_type == "Both" and has_spring_data):
            df_spring_stats = df_display[df_display['TestType'] == 'Spring'].copy() if 'TestType' in df_display.columns else df_display.copy()

            if len(df_spring_stats) > 0:
                st.write("**Spring Launcher: Distance Statistics by Force Range**")
                force_ranges = [(2, 4), (4, 6), (6, 8)]
                stats_data = []

                for low, high in force_ranges:
                    mask = (df_spring_stats['Force'] >= low) & (df_spring_stats['Force'] < high)
                    if mask.sum() > 0:
                        subset = df_spring_stats[mask]['Distance']
                        stats_data.append({
                            'Force Range': f'{low}N-{high}N',
                            'Count': mask.sum(),
                            'Mean Distance': f"{subset.mean():.3f}m",
                            'Max Distance': f"{subset.max():.3f}m",
                            'Min Distance': f"{subset.min():.3f}m",
                            'Std Dev': f"{subset.std():.3f}m"
                        })

                if stats_data:
                    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

    with tab3:
        # Filter option
        col1, col2 = st.columns(2)
        with col1:
            search_q = st.text_input("Filter by value")

        # Apply filters
        display_df = df_display.copy()

        if search_q:
            # Search across all columns
            display_df = display_df[
                display_df.apply(lambda row: search_q.lower() in str(row).lower(), axis=1)]

        # Select relevant columns based on test type
        if view_test_type == "Gas Pump Tests":
            cols_to_show = ['Test', 'Trial', 'Angle', 'Distance']
        elif view_test_type == "Spring Launcher Tests":
            cols_to_show = ['Test', 'Trial', 'Force', 'SpringK', 'Distance']
        else:  # Both
            cols_to_show = ['Test', 'Trial', 'TestType', 'Angle', 'Force', 'SpringK', 'Distance']

        # Only show columns that exist
        cols_to_show = [c for c in cols_to_show if c in display_df.columns]

        st.dataframe(display_df[cols_to_show], use_container_width=True)
