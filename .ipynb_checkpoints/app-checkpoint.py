import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

from utils import simulate_gates, simulate_gates_over_time, objective, optimize_gate_openings, hybrid_loss_fn, smart_optimize_gates
from plot import plot_gates

st.set_page_config(page_title="Smart Water Manager", layout="wide")
# Initialize state only once
if "gates" not in st.session_state:
    st.session_state.gates = ["บรมธาตุ", "ชันสูตร", "บางระจัน", "ยางมณี", "ผักไห่"]

if "gate_levels" not in st.session_state:
    st.session_state.gate_levels = {gate: 0.2 for gate in st.session_state.gates}

# Sidebar: User Inputs
# st.sidebar.title("เลือกประตูระบายน้ำ")
# Sidebar navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["โหมดทดลอง (What-If)", "โหมดอัตโนมัติ (AI Mode)"])
st.sidebar.markdown("---")
# Generate time options: "00:00", "01:00", ..., "23:00"

# Set seed for reproducibility
random.seed(42)

# Generate time options
time_options = [f"{hour:02d}:00" for hour in range(24)]

interval_options = [hour for hour in range(0,25)]
interval_labels = ["ปัจจุบัน"] + [f"+{hour} ชั่วโมง" for hour in range(1,25)]
interval_map = {interval_labels[i]:interval_options[i] for i in range(len(interval_options))}

# Create mock dictionary with reproducible random values around 120
inflows = {time: random.randint(100, 140) for time in time_options}

# Sidebar dropdown
selected_time = st.sidebar.selectbox("เลือกเวลา", time_options, index=13)
predict_interval_label = st.sidebar.selectbox("เลือกเวลาทำนาย", interval_labels, index=2)
prediction_interval= interval_map[predict_interval_label]

inflow = inflows[selected_time]
# Checkbox to toggle manual input
manual_input = st.sidebar.checkbox("ใส่ค่าอัตราการไหลเอง")

# If checked, show the number input
if manual_input:
    inflow = st.sidebar.number_input("ป้อนค่าอัตราการไหล (cms)", min_value=0.0, step=0.1, value=float(inflow))

# Display the flow rate info
st.sidebar.markdown(
    f"""
    #### อัตราการไหลจากประตู C2  
    ณ เวลา {selected_time} น. = <span style='color:deepskyblue; font-weight:bold'>{inflow:.2f} cms</span>
    """,
    unsafe_allow_html=True
)
if page == "โหมดทดลอง (What-If)":
    # Title
    # st.markdown("""
    #     <div style="background-color:#0066cc; padding:10px 20px; border-radius:0px 0px 10px 10px">
    #         <h3 style="color:white; margin:0;">💧 ระบบบริหารจัดการน้ำอัจฉริยะ</h3>
    #     </div>
    # """, unsafe_allow_html=True)
    # st.title("📊 ระบบแดชบอร์ดบริหารจัดการน้ำอัจฉริยะ")
    st.title("โหมดทดลอง (What-If)")
    # selected_gate = st.selectbox("เลือกประตูระบายน้ำ", ["มโนรมย์", "ช่องแค", "โคกกระเทียม", "เริงราง"])
    st.markdown("---")

    # 1️⃣ Create placeholder to push plot to the top
    lspace, chart, rspace = st.columns([1,20,1])
    with chart:
        plot_placeholder = st.empty()
    st.markdown("---")

    # 2️⃣ Layout - Inputs Below the Plot
    col1, col_sep, col2 = st.columns([2, 0.2, 5])

    with col1:
        st.markdown("#### กำหนดระดับน้ำ (ม.)")
        # inflow = st.number_input("อัตราการไหลจากประตู C2", min_value=0.0, max_value=500.0, value=120.0, step=1.0, key="inflow_input")
        col_min, col_max = st.columns(2)
        with col_min:
            min_w_height = st.number_input("ค่าต่ำสุด (m)", min_value=0.0, max_value=50.0, value=6.0, step=0.1)
        with col_max:
            max_w_height = st.number_input("ค่าสูงสุด (m)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
            
        
    with col_sep:
        st.markdown(
            """
            <style>
            .vertical-line {
                height: 500px;
                border-left: 2px solid #D3D3D3;
                margin: auto;
            }
            </style>
            <div class="vertical-line"></div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # current_ys
        st.markdown("#### ระดับน้ำปัจจุบัน (ม.)")
        cols = st.columns(len(st.session_state.gates)+1)
        st.session_state.water_levels = {st.session_state.gates[i]:11-i for i in range(len(st.session_state.gates))}
        for i in range(len(st.session_state.gates)):
            with cols[i]:
                gate = st.session_state.gates[i]
                st.session_state.water_levels[gate] = st.number_input(
                    f"ประตู{gate}",
                    min_value=0.1,
                    max_value=100.,
                    value=float(st.session_state.water_levels[gate]),
                    step=0.1,
                    key=f"y_input_{gate}"
            )
                
        st.markdown("#### ปรับระดับความสูงของประตู (ม.)")
        cols = st.columns(len(st.session_state.gates)+1)
        for i in range(len(st.session_state.gates)):
            with cols[i]:
                gate = st.session_state.gates[i]
                st.session_state.gate_levels[gate] = st.number_input(
                    f"ประตู{gate}",
                    min_value=0.1,
                    max_value=2.0,
                    value=st.session_state.gate_levels[gate],
                    step=0.1,
                    key=f"gate_input_{gate}"
            )
        with cols[len(st.session_state.gates)]:
            st.markdown(" ")  # Spacer ให้ปุ่มอยู่ตรงแนวเดียวกัน
            st.markdown(" ")  # Spacer ให้ปุ่มอยู่ตรงแนวเดียวกัน
            if st.button("รีเซ็ต"):
                for i in range(len(st.session_state.gates)):
                    gate = st.session_state.gates[i]
                    st.session_state.gate_levels[gate] = 0.2
                st.rerun()

        
    # 3️⃣ After all inputs are defined, simulate and draw
    Cds = [0.5]*len(st.session_state.gates)
    gate_levels = [st.session_state.gate_levels[gate] for gate in st.session_state.gates]
    if prediction_interval!=0:
        dt = 10
        initial_ys=list(st.session_state.water_levels.values())
        current_levels = initial_ys
        qs, water_levels = simulate_gates_over_time(inflow, gate_levels, initial_ys=initial_ys, Cds=Cds, dt=dt, steps=prediction_interval*3600//dt)
        water_levels = np.concatenate([[max_w_height], water_levels[-1]])
        qs=np.concatenate([[inflow], qs[-1,1:]])
    else:
        current_levels = list(st.session_state.water_levels.values())
        qs, _ = simulate_gates(inflow, gate_levels, initial_y0=max_w_height, Cds=Cds)
        water_levels = [max_w_height]+list(st.session_state.water_levels.values())
    # print(water_levels)
    gate_names = ["C2", "Boromthat", "Chanasut", "Bangrajan", "Yangmani", "Pak-hai"]
    gate_heights = [0] + gate_levels
    gate_positions = [i for i in range(len(st.session_state.gates)+1)]

    

    fig = plot_gates(gate_names, gate_heights, gate_positions, water_levels, qs, y_min=min_w_height, y_max=max_w_height, current_levels=current_levels)
    st.session_state.fig = fig
    # fig = animate_water_levels(gate_names, gate_heights, gate_positions, water_levels, qs, y_min=min_w_height, y_max=max_w_height)
    # Generate and display


    # 4️⃣ Fill the top placeholder with the plot
    with plot_placeholder:
        st.markdown("### 🧭 ภาพจำลองประตู")
        st.pyplot(fig)
        # gif_path = animate_ripple_color(gate_names, gate_heights, gate_positions, water_levels, qs, "water.gif")
        # st.image(gif_path)


elif page == "โหมดอัตโนมัติ (AI Mode)":  

    # Main Panel
    # st.title("📊 ระบบแดชบอร์ดบริหารจัดการน้ำอัจฉริยะ")
    st.title(f"โหมดอัตโนมัติ (AI Mode)")
    # 1️⃣ Create placeholder to push plot to the top
    # selected_gate = st.selectbox("เลือกประตูระบายน้ำ", ["มโนรมย์", "ช่องแค", "โคกกระเทียม", "เริงราง"])
    st.markdown("---")
    
    Cds = [0.5]*len(st.session_state.gates)
    gate_levels = [st.session_state.gate_levels[gate] for gate in st.session_state.gates]

    col1, col2 = st.columns([5,1])

    with col1:
        plot_placeholder = st.empty()
    
    with col2:
        st.markdown("### เงื่อนไข")
        max_w_height = st.number_input("ระดับน้ำสูงสุด (m)", min_value=0.0, max_value=50.0, value=12.0, step=0.1)
        min_w_height = st.number_input("ระดับน้ำต่ำสุด (m)", min_value=0.0, max_value=50.0, value=6.0, step=0.1)
        q_target = st.number_input("อัตราการไหลปลายน้ำ", value=12.)
        ys = None
        # Data
        sections = ["Section A", "Section B", "Section C"]
        gate_names = ["C2", "Boromthat", "Chanasut", "Bangrajan", "Yangmani", "Pak-hai"]
        gate_levels = [st.session_state.gate_levels[gate] for gate in st.session_state.gates]
        gate_heights = [100]+[i for i in gate_levels]
        gate_positions = [i for i in range(len(st.session_state.gates)+1)]

    st.markdown("#### ระดับน้ำปัจจุบัน (ม.)")
    cols = st.columns(len(st.session_state.gates)+1)
    if "water_levels" not in st.session_state:
        st.session_state.water_levels = {st.session_state.gates[i]:11-i for i in range(len(st.session_state.gates))}
    for i in range(len(st.session_state.gates)):
        with cols[i]:
            gate = st.session_state.gates[i]
            st.session_state.water_levels[gate] = st.number_input(
                f"ประตู{gate}",
                min_value=0.1,
                max_value=100.,
                value=float(st.session_state.water_levels[gate]),
                step=0.1,
                key=f"y_input_{gate}")
    st.markdown("#### ระดับความสูงของประตู (ม.)")
    cols = st.columns(len(st.session_state.gates)+1)
    for i in range(len(st.session_state.gates)):
        with cols[i]:
            gate = st.session_state.gates[i]
            st.session_state.gate_levels[gate] = st.number_input(
                f"ประตู{gate}",
                min_value=0.1,
                max_value=2.0,
                value=st.session_state.gate_levels[gate],
                step=0.1,
                key=f"gate_input_{gate}",
                disabled=True
        )
    with col2:
        if prediction_interval and st.button("ปรับอัตโนมัติ"):
            dt = 10
            # Example usage:
            best_h, loss, (q_vals, y_vals) = smart_optimize_gates(q0=inflow, q_target=q_target, Cds=Cds, initial_ys=list(st.session_state.water_levels.values()), y_min=min_w_height, y_max=max_w_height, y_target=ys, dt=dt, steps=prediction_interval*3600//dt)
            qs=np.concatenate([[inflow], q_vals[-1,1:]])
            water_levels = np.concatenate([[max_w_height], y_vals[-1]])
            # water_levels = y_vals[-1]
            gate_heights = [100] + best_h.tolist()
            for i in range(len(st.session_state.gates)):
                gate = st.session_state.gates[i]
                st.session_state.gate_levels[gate] = gate_heights[i+1]
    
            st.rerun()
            # 4️⃣ Fill the top placeholder with the plot
            with plot_placeholder:
                fig = plot_gates(gate_names, gate_heights, gate_positions, water_levels, qs, y_min=min_w_height, y_max=max_w_height)
                st.markdown("### 🧭 ภาพจำลองประตู")
                st.pyplot(fig)
        elif prediction_interval:
            dt = 10
            initial_ys=list(st.session_state.water_levels.values())
            current_levels = initial_ys
            qs, water_levels = simulate_gates_over_time(inflow, gate_levels, initial_ys=initial_ys, Cds=Cds, dt=dt, steps=prediction_interval*3600//dt)
            water_levels = np.concatenate([[max_w_height], water_levels[-1]])
            qs=np.concatenate([[inflow], qs[-1,1:]])
            gate_names = ["C2", "Boromthat", "Chanasut", "Bangrajan", "Yangmani", "Pak-hai"]
            gate_heights = [0] + gate_levels
            gate_positions = [i for i in range(len(st.session_state.gates)+1)]
           
        else:
            qs, _ = simulate_gates(inflow, gate_levels, initial_y0=max_w_height, Cds=Cds)
            water_levels = [max_w_height]+list(st.session_state.water_levels.values())
        with plot_placeholder:
            fig = plot_gates(gate_names, gate_heights, gate_positions, water_levels, qs, y_min=min_w_height, y_max=max_w_height, current_levels=current_levels)
            st.pyplot(fig)
    

