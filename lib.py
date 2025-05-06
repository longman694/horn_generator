from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from plotly.subplots import make_subplots


__all__ = (
    'create_2d_plot', 'create_3d_plot',
    'plot_demo', 'interpolate', 'generate_hcd_horn', 'generate_tractrix_horn',
    'generate_spherical_horn', 'generate_exponential_horn'
)


def create_2d_plot(x, y):
    # 2D Plot
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Horn'))
    fig2d.update_layout(title='2D Horn Profile',
                        xaxis=dict(range=[0, None], scaleanchor='y'),
                        yaxis=dict(range=[0, None], scaleanchor='x'),
                        xaxis_title='x (mm)',
                        yaxis_title='Radius y (mm)',
                        height=600, width=800)
    return fig2d


def create_3d_plot(x, y):
    # 3D wireframe (rotate profile)
    theta = np.linspace(0, 2*np.pi, 50)
    X = []
    Y = []
    Z = []

    for i in range(len(x)):
        X.extend([x[i]] * len(theta))
        Y.extend(y[i] * np.cos(theta))
        Z.extend(y[i] * np.sin(theta))

    fig3d = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='lines+markers',
        marker=dict(size=1),
        name='3D Horn'
    )])

    fig3d.update_layout(title='3D Horn (Wireframe)',
                        scene=dict(
                            xaxis_title='x (mm)',
                            yaxis_title='y (mm)',
                            zaxis_title='z (mm)',
                            aspectmode="data",
                            aspectratio=dict(x=1, y=1, z=1),
                        ),
                        height=800, width=800)
    return fig3d


def plot_demo(x, y):
    """
    plot the horn in 2D and 3D
    x and y are np.array in millimeters
    """
    fig2d = create_2d_plot(x, y)
    fig2d.show()

    fig3d = create_3d_plot(x, y)
    fig3d.show()


def interpolate(df: pd.DataFrame, num_point=10) -> pd.DataFrame:
    max_x = df['x (mm)'].max()
    new_df = pd.DataFrame({'x (mm)': np.linspace(0, max_x, num_point)})

    spl = CubicSpline(df['x (mm)'], df['y (mm)'])
    new_df['y (mm)'] = spl(new_df['x (mm)'])
    return new_df


def generate_tractrix_horn(throat_radius, cutoff_freq, num_points=10, plot=True):
    """
    throat_radius in mm
    cutoff freq in Hz
    num_points in number of points
    """
    throat_radius /= 1000  # convert to meters
    c = 343.0  # m/s
    a = c / (2 * np.pi * cutoff_freq)

    y = np.linspace(throat_radius, a, num_points)
    x = a * np.log((a + np.sqrt(a**2 - y**2)) / y) - np.sqrt(a**2 - y**2)

    df = pd.DataFrame({'x (m)': x, 'y (m)': y})

    df['x (m)'] = np.abs(df['x (m)'] - df['x (m)'].max())  # flip the profile

    df['x (mm)'] = df['x (m)'] * 1000
    df['y (mm)'] = df['y (m)'] * 1000

    if plot:
        plot_demo(df['x (mm)'], df['y (mm)'])

    return df[['x (mm)', 'y (mm)']]


def generate_spherical_horn(throat_radius, cutoff_freq, scale=4, fold=False, fold_back=True, plot=True):
    """
    throat_radius in mm
    cutoff freq in Hz
    scale resolution in mm
    fold `True` for folding horn backward (default: False)
    fold_back `True` allow to fold back beyond the tweeter location (default: True)
    """
    throat_radius /= 1000  # convert to meters
    scale /= 1000
    c = 343.0  # m/s
    r0 = c / np.pi / cutoff_freq
    h0 = r0 - np.sqrt(r0*r0 - throat_radius*throat_radius)

    flare_rate = 4 * np.pi * cutoff_freq / c  # m

    x = np.arange(0, 1, scale)
    h = h0 * np.exp(flare_rate * x)
    xh = x - h + h0
    s = 2 * np.pi * r0 * h
    df = pd.DataFrame({'x': x, 'h': h, 'xh': xh, 's': s})
    if not fold:
        max_xh = df['xh'].max()
        max_x = df[df['xh'] == max_xh]['x'].max()
        df = df[df['x']<=max_x]
    else:
        df = df[df['s']/np.pi - df['h']**2 >= 0]
        if not fold_back:
            df = df[df['xh']>=0]

    df['y'] = np.sqrt(df['s'] / np.pi - df['h']**2)

    df['x (mm)'] = df['xh'] * 1000
    df['y (mm)'] = df['y'] * 1000

    if plot:
        plot_demo(df['x (mm)'], df['y (mm)'])

    return df[['x (mm)', 'y (mm)']]


def generate_exponential_horn(throat_radius, cutoff_freq, scale=4, plot=True):
    """
    throat_radius in mm
    cutoff freq in Hz
    scale resolution in mm
    """
    throat_radius /= 1000  # convert to meters
    scale /= 1000
    c = 343.0

    wave_length = c / cutoff_freq
    growth_factor = 4 * np.pi / wave_length

    x = np.arange(0, 1, scale)
    s = throat_radius**2 * np.pi * np.exp(growth_factor * x)
    r = np.sqrt(s/np.pi)
    cir = 2 * np.pi * r
    krm = cir / wave_length

    df = pd.DataFrame({'x':x, 'y': r, 'krm': krm})
    df = df[df['krm'] <= 1]

    df['x (mm)'] = df['x'] * 1000
    df['y (mm)'] = df['y'] * 1000

    if plot:
        plot_demo(df['x (mm)'], df['y (mm)'])

    return df[['x (mm)', 'y (mm)']]


def generate_hcd_horn(origin_profile: pd.DataFrame, mouth_ratio=1.7, mode: Literal['linear', 'para', 'exp', 'log', 'hyper', 'logistic'] = 'linear', acc=1.0, plot=True) -> tuple[pd.DataFrame, list[go.Figure]]:
    """
    origin_profile: pd.DataFrame
    mode: 'linear', 'para', 'exp', 'log', 'hyper', 'logistic' more https://www.desmos.com/calculator/e2csdxezor
    """
    df = origin_profile.copy()
    df['area'] = np.pi * df['y (mm)']**2

    org_mouth_ratio = mouth_ratio
    if acc > 1.0:
        mouth_ratio *= acc

    if mode == 'linear':
        mouth_ratio_transform = CubicSpline([0, df['x (mm)'].max()], [1, mouth_ratio])

    elif mode == 'para':
        mr = df[['x (mm)']].copy()
        max_x = mr['x (mm)'].max()
        mr['_x'] = CubicSpline([0, max_x], [0,1])(mr['x (mm)'])
        mr['y'] = mr['_x']**2 * (mouth_ratio - 1) + 1
        max_index = mr[mr['x (mm)'] == max_x].index.min()
        mouth_ratio_transform = CubicSpline(mr[mr.index <= max_index]['x (mm)'], mr[mr.index <= max_index]['y'])

    elif mode == 'exp':
        mr = df[['x (mm)']].copy()
        max_x = mr['x (mm)'].max()
        mr['_x'] = CubicSpline([0, max_x], [0,1])(mr['x (mm)'])
        mr['y'] = np.sqrt(mr['_x']) * (mouth_ratio - 1) + 1
        max_index = mr[mr['x (mm)'] == max_x].index.min()
        mouth_ratio_transform = CubicSpline(mr[mr.index <= max_index]['x (mm)'], mr[mr.index <= max_index]['y'])

    elif mode == 'log':
        mr = df[['x (mm)']].copy()
        max_x = mr['x (mm)'].max()
        mr['_x'] = CubicSpline([0, max_x], [0,1])(mr['x (mm)'])
        mr['y'] = np.log10(9*mr['_x'] + 1) * (mouth_ratio - 1) + 1
        max_index = mr[mr['x (mm)'] == max_x].index.min()
        mouth_ratio_transform = CubicSpline(mr[mr.index <= max_index]['x (mm)'], mr[mr.index <= max_index]['y'])

    elif mode == 'hyper':
        mr = df[['x (mm)']].copy()
        max_x = mr['x (mm)'].max()
        mr['_x'] = CubicSpline([0, max_x], [0,1])(mr['x (mm)'])
        mr['y'] = np.sqrt((mr['_x']+1)**2 - 1)/np.sqrt(3) * (mouth_ratio - 1) + 1
        max_index = mr[mr['x (mm)'] == max_x].index.min()
        mouth_ratio_transform = CubicSpline(mr[mr.index <= max_index]['x (mm)'], mr[mr.index <= max_index]['y'])

    elif mode == 'logistic':
        mr = df[['x (mm)']].copy()
        max_x = mr['x (mm)'].max()
        mr['_x'] = CubicSpline([0, max_x], [0,1])(mr['x (mm)'])
        mr['y'] = 1/(1 + np.exp(11*(0.5-mr['_x']))) * (mouth_ratio - 1) + 1
        max_index = mr[mr['x (mm)'] == max_x].index.min()
        mouth_ratio_transform = CubicSpline(mr[mr.index <= max_index]['x (mm)'], mr[mr.index <= max_index]['y'])

    else:
        raise ValueError('`mode` must be `linear`, `para`, `exp`, `log`, or `hyper`')

    # df['mouth_ratio'] = mouth_ratio
    df['mouth_ratio'] = mouth_ratio_transform(df['x (mm)']).clip(max=org_mouth_ratio)
    first_max_index = df[df['mouth_ratio'] == df['mouth_ratio'].max()].index.min()
    df.loc[df.index > first_max_index, 'mouth_ratio'] = org_mouth_ratio

    df['b'] = np.sqrt(df['area']/ np.pi / df['mouth_ratio'])
    df['a'] = df['b'] * df['mouth_ratio']

    fig_transition = go.Figure()
    fig_transition.add_trace(go.Scatter(x=df.index, y=df['mouth_ratio'], mode='lines', name='Mouth Ratio'))
    fig_transition.update_layout(title='Mouth ratio',
                        xaxis_title='point',
                        yaxis_title='mouth ratio',
                        height=600, width=1200)

    fig2d = make_subplots(rows=1, cols=1, subplot_titles=("Comparison of Circular and HCD Horn Profiles",))

    fig2d.add_trace(go.Scatter(x=df['x (mm)'], y=df['y (mm)'], mode='lines', name='Circular Horn Profile', line=dict(dash='dash')), row=1, col=1)
    fig2d.add_trace(go.Scatter(x=df['x (mm)'], y=df['b'], mode='lines', name='HCD Horn Profile (Radius 1)'), row=1, col=1)
    fig2d.add_trace(go.Scatter(x=df['x (mm)'], y=df['a'], mode='lines', name='HCD Horn Profile (Radius 2)'), row=1, col=1)

    fig2d.update_layout(
        title='2D HCD Horn',
        title_font=dict(size=20),
        showlegend=True,
        xaxis=dict(range=[0, None], scaleanchor='y'),
        yaxis=dict(range=[0, None], scaleanchor='x'),
        xaxis_title='x (mm)',
        yaxis_title='Radius y (mm)',
        height=600, width=1200
    )

    theta = np.linspace(0, 2*np.pi, 40)

    # Lists to hold ellipse points
    x_vals, y_vals, z_vals = [], [], []

    for i, row in df.iterrows():
        x0 = row['x (mm)']
        a = row['a']
        b = row['b']

        # Parametrize ellipse in XY plane at z=0
        x_ellipse = a * np.cos(theta)
        y_ellipse = b * np.sin(theta)
        z_ellipse = np.full_like(theta, x0)  # separate each ellipse in z for visibility

        x_vals.extend(x_ellipse)
        y_vals.extend(y_ellipse)
        z_vals.extend(z_ellipse)

    # Plotting
    fig3d = go.Figure(data=go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='lines+markers',
        marker=dict(size=1),
        line=dict(width=1),
    ))

    fig3d.update_layout(
        title="3D HCD Horn",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (ellipse index)",
            aspectmode='data',  # maintains equal scaling
            aspectratio=dict(x=1, y=1, z=1),
        ),
        height=800, width=1200,
    )

    if plot:
        fig_transition.show()
        fig2d.show()
        fig3d.show()

    return df[['x (mm)', 'y (mm)', 'a', 'b']], [fig_transition, fig2d, fig3d]
