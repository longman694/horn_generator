import io

import streamlit as st
import pandas as pd
import numpy as np

from lib import *

with st.sidebar:
    st.title('Horn Generator')

    horn_type = st.selectbox(
        'Select Horn type',
        options=['Tractrix', 'Spherical', 'Exponential'],
        index=0
    )

    st.text(f'Horn type: {horn_type}')

    throat_r = st.number_input('Throat Radius (mm)', value=15.0, min_value=1.0, max_value=1000.0)
    cutoff_f = st.number_input('Cutoff Frequency (Hz)', value=1000, min_value=1, max_value=20000)

    df = pd.DataFrame()
    fold = False
    if horn_type == 'Tractrix':
        num_points = st.number_input('Number of points', value=10, min_value=1, max_value=1000)

        df = generate_tractrix_horn(throat_r, cutoff_f, num_points, plot=False)

    elif horn_type == 'Spherical':
        scale = st.number_input('Scale resolution (mm)', value=4.0, min_value=0.1, max_value=20.0)
        fold = st.checkbox('allow to fold', value=False)
        fold_back = st.checkbox('allow to fold back', value=True, disabled=not fold)

        df = generate_spherical_horn(throat_r, cutoff_f, scale, fold, fold_back, plot=False)

    elif horn_type == 'Exponential':
        scale = st.number_input('Scale resolution (mm)', value=4.0, min_value=0.1, max_value=20.0)
        df = generate_exponential_horn(throat_r, cutoff_f, scale, plot=False)

    enable_hcd = st.checkbox('HCD', value=False, disabled=fold)

buffer = io.BytesIO()

if not enable_hcd:
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    st.download_button(
        label='export to excel',
        data=buffer.getvalue(),
        file_name=f'{horn_type.capitalize()}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.plotly_chart(create_2d_plot(df['x (mm)'], df['y (mm)']), use_container_width=True)
    st.plotly_chart(create_3d_plot(df['x (mm)'], df['y (mm)']), use_container_width=True)
    st.table(df)

if enable_hcd:
    with st.sidebar:
        mouth_ratio = st.number_input('Mouth Ratio', value=1.7, min_value=1.0, max_value=2.0)
        mode = st.selectbox(
            'HCD mode',
            options=['linear', 'para', 'exp', 'log', 'hyper', 'logistic', 'acc'],
            index=0
        )
        acc = st.number_input('Accelerate', value=1.0, min_value=1.0, max_value=1.5)

    hcd, figs = generate_hcd_horn(df, mouth_ratio, mode, acc, plot=False)

    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        hcd.to_excel(writer, sheet_name='Sheet1', index=False)
    st.download_button(
        label='export to excel',
        data=buffer.getvalue(),
        file_name=f'{horn_type.capitalize()}_HCD_{mouth_ratio:.1f}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    for fig in figs:
        st.plotly_chart(fig, use_container_width=True)

    st.table(hcd)
