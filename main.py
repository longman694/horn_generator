import time

import pandas as pd
import streamlit as st

from lib import *

with st.sidebar:
    st.title('Horn Generator')

    horn_type = st.selectbox(
        'Select Horn type',
        options=['Tractrix', 'Spherical', 'Exponential'],
        index=0
    )

    throat_r = st.number_input('Throat Radius (mm)', value=15.0, min_value=1.0, max_value=1000.0, step=1.0)
    cutoff_f = st.number_input('Cutoff Frequency (Hz)', value=1000, min_value=10, max_value=15000, step=100)

    df = pd.DataFrame()
    fold = False

    if horn_type == 'Tractrix':
        num_points = st.number_input('Number of points', value=10, min_value=1, max_value=50)

        df = generate_tractrix_horn(throat_r, cutoff_f, num_points, plot=False)

    elif horn_type == 'Spherical':
        scale = st.number_input('Scale resolution (mm)', value=4.0, min_value=0.5, max_value=20.0, step=1.0)
        fold = st.checkbox('allow to fold', value=False)
        fold_back = st.checkbox('allow to fold back', value=True, disabled=not fold)

        df = generate_spherical_horn(throat_r, cutoff_f, scale, fold, fold_back, plot=False)

    elif horn_type == 'Exponential':
        scale = st.number_input('Scale resolution (mm)', value=4.0, min_value=0.5, max_value=20.0, step=1.0)
        df = generate_exponential_horn(throat_r, cutoff_f, scale, plot=False)

    st.divider()

    enable_hcd = st.checkbox('HCD', value=False, help="Enable HCD (Hybrid Constant Directivity) mode")

download_render = st.form('download_render_form', border=False)
submit = download_render.form_submit_button('Download')

if not enable_hcd:
    with st.sidebar:
        st.divider()
        step_edge_width = st.number_input(
            'Edge width (mm) for STEP file', value=1.0, min_value=1.0, max_value=10.0, step=0.5,
            help="Try to increase this if the solid from STEP file's broken"
        )

    if submit:
        with st.container(border=True):
            with st.spinner("rendering ...", show_time=True):
                time.sleep(0.8)
                excel_data = generate_excel(df)
                dxf_data = generate_dxf(df)
                step_data = None
                if not fold or (fold and not fold_back):
                    step_data = generate_step(df, False, fold, step_edge_width)

                left, middle, right = st.columns(3)
                left.download_button(
                    label='Export to Excel',
                    data=excel_data,
                    file_name=f'{horn_type.capitalize()}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
                middle.download_button(
                    label='Export to DXF',
                    data=dxf_data,
                    file_name=f'{horn_type.capitalize()}.dxf',
                    mime='model/vnd.dwf',
                    use_container_width=True
                )
                if not fold or (fold and not fold_back):
                    right.download_button(
                        label='Export to STEP',
                        data=step_data,
                        file_name=f'{horn_type.capitalize()}.step',
                        mime='application/STEP',
                        use_container_width=True
                    )
                else:
                    right.button(
                        label='Export to STEP',
                        disabled=True,
                        help='Not support for fold back horn.',
                        use_container_width=True
                    )

    st.plotly_chart(create_2d_plot(df['x (mm)'], df['y (mm)']), use_container_width=True)
    st.plotly_chart(create_3d_plot(df['x (mm)'], df['y (mm)']), use_container_width=True)
    st.table(df)

if enable_hcd:
    with st.sidebar:
        mouth_ratio = st.number_input('Mouth Ratio', value=1.7, min_value=1.0, max_value=3.0, step=0.1)
        mode = st.selectbox(
            'HCD mode',
            options=['linear', 'para', 'exp', 'log', 'hyper', 'logistic'],
            index=0
        )
        acc = st.number_input('Accelerate', value=1.0, min_value=1.0, max_value=2.0, step=0.1)

        st.divider()
        step_edge_width = st.number_input(
            'Edge width (mm) for STEP file', value=2.0, min_value=1.0, max_value=10.0, step=0.5,
            help="Try to increase this if the solid from STEP file's broken"
        )

    hcd, figs = generate_hcd_horn(df, mouth_ratio, mode, acc, plot=False)

    if submit:
        with st.container(border=True):
            with st.spinner("rendering ...", show_time=True):
                time.sleep(0.8)
                excel_data = generate_excel(hcd)
                dxf_data = None
                step_data = None
                if not fold or (fold and not fold_back):
                    step_data = generate_step(hcd, True, fold, step_edge_width)

                left, middle, right = st.columns(3)
                left.download_button(
                    label='Export to Excel',
                    data=excel_data,
                    file_name=f'{horn_type.capitalize()}_HCD_{mouth_ratio:.1f}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True,
                )
                middle.button(
                    label='Export to DXF',
                    disabled=True,
                    help='Only available on circular mode.',
                    use_container_width=True
                )
                if not fold or (fold and not fold_back):
                    right.download_button(
                        label='Export to STEP',
                        data=step_data,
                        file_name=f'{horn_type.capitalize()}_HCD_{mouth_ratio:.1f}.step',
                        mime='application/STEP',
                        use_container_width=True
                    )
                else:
                    right.button(
                        label='Export to STEP',
                        disabled=True,
                        help='Not support for fold back horn.',
                        use_container_width=True
                    )
    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

    st.table(hcd)

st.divider()
footer_markdown = """
<div style="text-align: center; color: #888; font-size: 0.9em;">
    Napat Charoenlarpkul's Horn Generator &copy; 2025 | <a href="https://github.com/longman694/horn_generator" target="_blank" rel="noopener noreferrer">Source Code</a>
</div>
"""
st.markdown(footer_markdown, unsafe_allow_html=True)