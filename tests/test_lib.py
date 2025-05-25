import pytest
import pandas as pd
import numpy as np
from lib import (
    generate_tractrix_horn,
    generate_spherical_horn,
    generate_exponential_horn,
    generate_hcd_horn,
    generate_excel,
    generate_dxf,
    generate_step,
    interpolate,
    create_2d_plot,
    create_3d_plot,
)
from io import BytesIO
import openpyxl # For reading Excel files
import ezdxf # For reading DXF files
import plotly.graph_objects as go
from pandas.testing import assert_frame_equal

# Helper function for generating sample data for HCD horn tests
@pytest.fixture
def sample_horn_data():
    # Using generate_tractrix_horn to create a base profile for HCD tests
    return generate_tractrix_horn(throat_radius=10, cutoff_freq=1000, num_points=20, plot=False)

# Tests for generate_tractrix_horn
def test_generate_tractrix_horn_typical_values():
    throat_radius = 15
    cutoff_freq = 1000
    num_points = 10
    df = generate_tractrix_horn(throat_radius, cutoff_freq, num_points, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['x (mm)', 'y (mm)'])
    assert len(df) == num_points
    assert np.isclose(df['y (mm)'].iloc[0], throat_radius, atol=1e-1) # Looser tolerance for floating point comparisons
    assert all(df['x (mm)'] >= 0)
    # Assert that x values are generally increasing (or at least not decreasing)
    assert all(np.diff(df['x (mm)']) >= -1e-9) # Using a small tolerance for floating point issues

def test_generate_tractrix_horn_min_points():
    df = generate_tractrix_horn(throat_radius=10, cutoff_freq=1000, num_points=2, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

# Tests for generate_spherical_horn
def test_generate_spherical_horn_typical_no_fold():
    throat_radius = 25
    cutoff_freq = 500
    scale = 4
    df = generate_spherical_horn(throat_radius, cutoff_freq, scale, fold=False, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['x (mm)', 'y (mm)'])
    assert np.isclose(df['y (mm)'].iloc[0], throat_radius, atol=throat_radius * 0.1 + 1) # Adjust tolerance based on expected value
    assert all(df['x (mm)'] >= -1e-9) # Allow for small floating point inaccuracies around zero

def test_generate_spherical_horn_fold_true():
    df = generate_spherical_horn(throat_radius=25, cutoff_freq=500, scale=4, fold=True, fold_back=True, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['x (mm)', 'y (mm)'])
    # More specific assertions for folded geometry can be complex;
    # for now, ensuring it runs and output has basic properties.
    assert len(df) > 0 # Ensure some data is generated

def test_generate_spherical_horn_fold_true_no_fold_back():
    df = generate_spherical_horn(throat_radius=25, cutoff_freq=500, scale=4, fold=True, fold_back=False, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['x (mm)', 'y (mm)'])
    assert all(df['x (mm)'] >= -1e-9) # x values should be non-negative when not folding back beyond tweeter

# Tests for generate_exponential_horn
def test_generate_exponential_horn_typical_values():
    throat_radius = 20
    cutoff_freq = 800
    scale = 4
    df = generate_exponential_horn(throat_radius, cutoff_freq, scale, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ['x (mm)', 'y (mm)'])
    assert np.isclose(df['y (mm)'].iloc[0], throat_radius, atol=1e-1)
    assert all(df['x (mm)'] >= 0)
    assert len(df) > 1 # Ensure more than one point is generated

# Tests for generate_hcd_horn
HCD_MODES = ['linear', 'para', 'exp', 'log', 'hyper', 'logistic']

@pytest.mark.parametrize("mode", HCD_MODES)
def test_generate_hcd_horn_modes(sample_horn_data, mode):
    df_hcd, figs = generate_hcd_horn(sample_horn_data, mouth_ratio=1.7, mode=mode, acc=1.0, plot=False)
    assert isinstance(df_hcd, pd.DataFrame)
    assert all(col in df_hcd.columns for col in ['x (mm)', 'y (mm)', 'a', 'b'])
    assert pd.api.types.is_numeric_dtype(df_hcd['a'])
    assert pd.api.types.is_numeric_dtype(df_hcd['b'])
    assert len(df_hcd) == len(sample_horn_data)
    assert isinstance(figs, list) # Check that a list of figures is returned

def test_generate_hcd_horn_invalid_mode(sample_horn_data):
    with pytest.raises(ValueError) as excinfo:
        generate_hcd_horn(sample_horn_data, mode="invalid_mode", plot=False)
    assert "`mode` must be" in str(excinfo.value)

def test_generate_hcd_horn_acc_effect(sample_horn_data):
    df_no_acc, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=1.5, acc=1.0, plot=False)
    df_acc, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=1.5, acc=1.2, plot=False)
    # With acc > 1, the initial transformation of mouth_ratio should be larger,
    # leading to potentially different 'a' and 'b' values, especially at the start.
    # This test just ensures the function runs with acc != 1.0
    assert not df_no_acc[['a', 'b']].equals(df_acc[['a', 'b']])

def test_generate_hcd_horn_plot_true(sample_horn_data):
    # Test that plot=True runs without error (actual plotting is not checked)
    # This may not be feasible in all CI environments if it tries to open GUI windows
    # For now, we rely on plot=False for most tests.
    # If generate_hcd_horn is modified to not show plots but return figures even with plot=True:
    df_hcd, figs = generate_hcd_horn(sample_horn_data, plot=True)
    assert isinstance(df_hcd, pd.DataFrame)
    assert isinstance(figs, list)
    assert len(figs) == 3 # Expecting transition, 2D, and 3D plots/figures
    # Further check if figs are plotly Figure objects if possible/needed
    # from plotly.graph_objects import Figure
    # assert all(isinstance(fig, Figure) for fig in figs)
    # This import and check might be too much if plotly isn't always a hard dep for lib.py users

# Test with a very small number of points for the input profile
@pytest.fixture
def minimal_horn_data():
    return generate_tractrix_horn(throat_radius=10, cutoff_freq=1000, num_points=3, plot=False)

def test_generate_hcd_horn_minimal_input(minimal_horn_data):
    df_hcd, _ = generate_hcd_horn(minimal_horn_data, mode='linear', plot=False)
    assert isinstance(df_hcd, pd.DataFrame)
    assert len(df_hcd) == 3
    assert all(col in df_hcd.columns for col in ['x (mm)', 'y (mm)', 'a', 'b'])

# Test edge case for mouth_ratio
def test_generate_hcd_horn_mouth_ratio_one(sample_horn_data):
    df_hcd, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=1.0, mode='linear', plot=False)
    assert isinstance(df_hcd, pd.DataFrame)
    # When mouth_ratio is 1, 'a' and 'b' should be equal to original 'y (mm)'
    # and mouth_ratio column in df should be 1 throughout
    # However, the internal 'mouth_ratio' column calculation might be complex.
    # A simpler check: 'a' and 'b' should be very close if mouth_ratio is 1.
    assert np.allclose(df_hcd['a'], df_hcd['b'])
    assert np.allclose(df_hcd['a'], sample_horn_data['y (mm)'])

# Test for generate_spherical_horn specific conditions
def test_generate_spherical_horn_low_cutoff():
    # Test with a low cutoff frequency, which might result in a large horn
    df = generate_spherical_horn(throat_radius=50, cutoff_freq=100, scale=10, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df['y (mm)'].iloc[0] > 0 # Radius should be positive

def test_generate_spherical_horn_high_cutoff():
    # Test with a high cutoff frequency
    df = generate_spherical_horn(throat_radius=10, cutoff_freq=5000, scale=1, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df['y (mm)'].iloc[0] > 0

# Test for generate_exponential_horn specific conditions
def test_generate_exponential_horn_large_throat():
    df = generate_exponential_horn(throat_radius=100, cutoff_freq=500, scale=10, plot=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert np.isclose(df['y (mm)'].iloc[0], 100, atol=1)

def test_generate_exponential_horn_krm_limit():
    # This function has a krm <=1 filter. Test that it's applied.
    # It's hard to predict the exact length without re-implementing the logic,
    # but we can check if it runs and produces a valid (possibly short) dataframe.
    df = generate_exponential_horn(throat_radius=10, cutoff_freq=200, scale=1, plot=False) # Low cutoff, should grow fast
    assert isinstance(df, pd.DataFrame)
    # If krm limit is hit early, df could be short. If not, it could be long.
    # Just ensure it's not empty and columns are correct.
    if not df.empty:
        assert all(col in df.columns for col in ['x (mm)', 'y (mm)'])
        assert all(df['x (mm)'] >= 0)
    # It's possible for the df to be empty if the first point itself has krm > 1
    # Depending on the exact calculation of krm for the first point (x=0)
    # For x=0, s = throat_radius^2 * pi. r = throat_radius.
    # cir = 2 * pi * throat_radius. wavelength = c / cutoff_freq
    # krm = (2 * pi * throat_radius) / (c / cutoff_freq)
    # krm = (2 * pi * throat_radius * cutoff_freq) / c
    # If this initial krm > 1, df will be empty.
    # Example: throat_radius=0.01m, cutoff_freq=2000Hz, c=343
    # krm = (2 * pi * 0.01 * 2000) / 343 = 125.66 / 343 = 0.36. This should not be empty.

    # Example that might result in empty or very short df:
    # High throat_radius, high cutoff_freq
    df_potentially_short = generate_exponential_horn(throat_radius=50, cutoff_freq=3000, scale=1, plot=False)
    assert isinstance(df_potentially_short, pd.DataFrame)
    # If it's empty, it means the first point already exceeded krm=1
    # krm_initial = (2 * np.pi * (50/1000) * 3000) / 343 = (2 * np.pi * 0.05 * 3000) / 343 = (0.1 * np.pi * 3000) / 343 = 942.47 / 343 = 2.74
    # Indeed, this krm > 1, so the dataframe should be empty.
    assert df_potentially_short.empty, "Expected empty DataFrame when initial krm > 1"

# Add a test to ensure plot=True also works for other generators (basic check)
def test_generate_tractrix_horn_plot_true():
    df = generate_tractrix_horn(10, 1000, 10, plot=True)
    assert isinstance(df, pd.DataFrame)

def test_generate_spherical_horn_plot_true():
    df = generate_spherical_horn(25, 500, plot=True)
    assert isinstance(df, pd.DataFrame)

def test_generate_exponential_horn_plot_true():
    df = generate_exponential_horn(20, 800, plot=True)
    assert isinstance(df, pd.DataFrame)

# Note: The `plot=True` tests are very basic. They primarily check that the functions
# run without error when `plot=True`. They do not verify the actual plot output, as that
# would require a more complex setup (e.g., mocking `plotly.graph_objects.Figure.show`).
# The `generate_hcd_horn` already has a `plot=True` test that checks for returned figures.
# For the other functions, `plot_demo` is called internally, which doesn't return figures.
# So, for those, we just check that they run.

# A more robust way for plot=True in generate_hcd_horn would be to mock fig.show()
# For now, the existing test for generate_hcd_horn with plot=True is:
# def test_generate_hcd_horn_plot_true(sample_horn_data):
#     df_hcd, figs = generate_hcd_horn(sample_horn_data, plot=True)
#     assert isinstance(df_hcd, pd.DataFrame)
#     assert isinstance(figs, list)
#     assert len(figs) == 3
# This seems fine as it checks the returned figures.
# The other functions (tractrix, spherical, exponential) call plot_demo which directly calls fig.show().
# Mocking fig.show() globally for those tests might be an option if issues arise in CI.
# Example using unittest.mock (if allowed and installed):
# from unittest.mock import patch
# @patch('plotly.graph_objects.Figure.show')
# def test_generate_tractrix_horn_plot_true_mocked(mock_show):
#     df = generate_tractrix_horn(10, 1000, 10, plot=True)
#     assert isinstance(df, pd.DataFrame)
#     mock_show.assert_called() # Or assert_has_calls if multiple plots

# For now, keeping the simpler plot=True tests.
# If they cause issues (e.g., trying to open GUI in headless CI), they might need adjustment
# or mocking as described above.

# Final check on HCD horn with acc and mouth_ratio interaction
def test_generate_hcd_horn_acc_and_mouth_ratio(sample_horn_data):
    df_hcd, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=2.0, acc=0.5, plot=False)
    assert isinstance(df_hcd, pd.DataFrame)
    # Max value in 'mouth_ratio' column of df_hcd should not exceed original mouth_ratio (2.0)
    # The internal transformation uses mouth_ratio * acc initially, but clips to original mouth_ratio
    # The 'mouth_ratio' column in the output df is the one *after* clipping.
    assert df_hcd['mouth_ratio'].max() <= 2.0 + 1e-9 # Add tolerance for float issues
    # And it should be greater than or equal to 1
    assert df_hcd['mouth_ratio'].min() >= 1.0 - 1e-9

    # Check if acc actually has an effect when it reduces the effective mouth_ratio for transformation
    df_base, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=2.0, acc=1.0, plot=False) # effective_mr = 2.0
    df_acc_reduce, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=2.0, acc=0.5, plot=False) # effective_mr for spline = 1.0

    # The resulting 'a' and 'b' should be different if acc modifies the transformation significantly
    # If acc=0.5 and mouth_ratio=2.0, the spline target is mouth_ratio*acc = 1.0, meaning almost no change from original.
    # So, 'a' and 'b' in df_acc_reduce should be very close to 'y (mm)'
    assert np.allclose(df_acc_reduce['a'], sample_horn_data['y (mm)'], atol=1e-1) # Looser tolerance
    assert np.allclose(df_acc_reduce['b'], sample_horn_data['y (mm)'], atol=1e-1)

    # And this should be different from df_base where acc=1.0 and mouth_ratio=2.0
    assert not np.allclose(df_base['a'], df_acc_reduce['a'])

# Ensure all x coordinates are positive or zero for tractrix horn
def test_generate_tractrix_horn_x_coordinates_positive():
    df = generate_tractrix_horn(throat_radius=10, cutoff_freq=500, num_points=100, plot=False)
    assert all(df['x (mm)'] >= -1e-9) # Check all x values are non-negative (allowing for float precision)

# Ensure y coordinates for tractrix horn are within expected bounds
def test_generate_tractrix_horn_y_coordinates_bounds():
    throat_radius = 10
    cutoff_freq = 500
    # a = c / (2 * np.pi * cutoff_freq) = 343 / (2 * pi * 500) approx 0.109m = 109mm
    # y ranges from throat_radius to 'a'
    df = generate_tractrix_horn(throat_radius=throat_radius, cutoff_freq=cutoff_freq, num_points=100, plot=False)
    c = 343.0
    a_val_mm = (c / (2 * np.pi * cutoff_freq)) * 1000
    assert df['y (mm)'].min() >= throat_radius - 1e-9
    assert df['y (mm)'].max() <= a_val_mm + 1e-9

# Ensure x coordinates for spherical horn are positive or zero when no fold
def test_generate_spherical_horn_x_coordinates_positive_no_fold():
    df = generate_spherical_horn(throat_radius=25, cutoff_freq=500, scale=4, fold=False, plot=False)
    assert all(df['x (mm)'] >= -1e-9)

# Ensure x coordinates for exponential horn are positive or zero
def test_generate_exponential_horn_x_coordinates_positive():
    df = generate_exponential_horn(throat_radius=20, cutoff_freq=800, scale=4, plot=False)
    if not df.empty: # Can be empty if initial krm > 1
        assert all(df['x (mm)'] >= -1e-9)

# Test for HCD horn that 'a' is always >= 'b'
def test_generate_hcd_horn_a_ge_b(sample_horn_data):
    df_hcd, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=1.5, plot=False)
    assert all(df_hcd['a'] >= df_hcd['b'] - 1e-9) # allow for floating point inaccuracies

    df_hcd_mr_less_one, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=0.7, plot=False)
    # if mouth_ratio < 1, then 'a' should be <= 'b'
    assert all(df_hcd_mr_less_one['a'] <= df_hcd_mr_less_one['b'] + 1e-9)

# Test for HCD horn that 'b' is related to original 'y (mm)' and mouth_ratio
# b = sqrt(area / (pi * mouth_ratio)) = sqrt(pi * y^2 / (pi * mouth_ratio)) = y / sqrt(mouth_ratio)
def test_generate_hcd_horn_b_relation(sample_horn_data):
    mouth_ratio_val = 1.7
    df_hcd, _ = generate_hcd_horn(sample_horn_data, mouth_ratio=mouth_ratio_val, plot=False)
    
    # Calculate expected 'b' based on the 'mouth_ratio' column in the output df,
    # as this column reflects the actual ratio applied at each point after transformation and clipping.
    expected_b = sample_horn_data['y (mm)'] / np.sqrt(df_hcd['mouth_ratio'])
    assert np.allclose(df_hcd['b'], expected_b, atol=1e-1, rtol=1e-2) # Looser tolerance due to multiple calculations

    # Also, 'a' should be b * mouth_ratio (using the df_hcd['mouth_ratio'] column)
    expected_a = df_hcd['b'] * df_hcd['mouth_ratio']
    assert np.allclose(df_hcd['a'], expected_a, atol=1e-1, rtol=1e-2)

# Fixtures for export tests
@pytest.fixture
def sample_export_df():
    data = {'x (mm)': [0, 1, 2, 3, 4, 5], 'y (mm)': [10, 12, 15, 13, 14, 16]} # Ensure enough points for spline in DXF
    return pd.DataFrame(data)

@pytest.fixture
def sample_hcd_export_df():
    data = {
        'x (mm)': [0, 1, 2, 3, 4, 5],
        'y (mm)': [10, 12, 15, 13, 14, 16], # y (mm) is present in HCD dataframes from generate_hcd_horn
        'a': [10, 11, 14, 12, 13, 15],
        'b': [8, 9, 12, 10, 11, 13]
    }
    return pd.DataFrame(data)

# Tests for generate_excel
def test_generate_excel(sample_export_df):
    excel_bytes = generate_excel(sample_export_df)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0
    
    # Read back the excel file
    df_read_back = pd.read_excel(BytesIO(excel_bytes))
    assert not df_read_back.empty
    pd.testing.assert_frame_equal(df_read_back, sample_export_df, check_dtype=False)

# Tests for generate_dxf
def test_generate_dxf(sample_export_df):
    dxf_bytes = generate_dxf(sample_export_df)
    assert isinstance(dxf_bytes, bytes)
    assert len(dxf_bytes) > 0
    
    # Try to load the DXF data
    try:
        doc = ezdxf.read(BytesIO(dxf_bytes))
        assert doc is not None
        # Check for a SPLINE entity (as generate_dxf creates one)
        msp = doc.modelspace()
        splines = list(msp.query('SPLINE'))
        assert len(splines) > 0, "No SPLINE entity found in DXF modelspace."
    except ezdxf.DXFStructureError as e:
        pytest.fail(f"Failed to read DXF: {e}")

# Tests for generate_step
def test_generate_step_normal(sample_export_df):
    step_bytes = generate_step(sample_export_df, hcd_enabled=False, fold=False)
    assert isinstance(step_bytes, bytes)
    assert len(step_bytes) > 0
    # Basic check for STEP file header/ender (very simplistic)
    assert b"HEADER;" in step_bytes
    assert b"ENDSEC;" in step_bytes
    assert b"END-ISO-10303-21;" in step_bytes

def test_generate_step_hcd(sample_hcd_export_df):
    step_bytes = generate_step(sample_hcd_export_df, hcd_enabled=True, fold=False)
    assert isinstance(step_bytes, bytes)
    assert len(step_bytes) > 0
    assert b"HEADER;" in step_bytes
    assert b"ENDSEC;" in step_bytes
    assert b"END-ISO-10303-21;" in step_bytes

def test_generate_step_fold(sample_export_df):
    # Create a profile that actually folds (x decreases)
    data_fold = {'x (mm)': [0, 1, 2, 1.5, 1, 0.5], 'y (mm)': [10, 12, 15, 13, 14, 16]}
    df_fold = pd.DataFrame(data_fold)
    step_bytes = generate_step(df_fold, hcd_enabled=False, fold=True)
    assert isinstance(step_bytes, bytes)
    assert len(step_bytes) > 0
    assert b"HEADER;" in step_bytes
    assert b"ENDSEC;" in step_bytes
    assert b"END-ISO-10303-21;" in step_bytes

# Test generate_step with HCD and fold enabled
def test_generate_step_hcd_fold(sample_hcd_export_df):
    # Create a profile that actually folds for HCD
    # For HCD, 'a' and 'b' columns are used.
    # We need x to decrease for folding to be meaningful in the test.
    data_hcd_fold = {
        'x (mm)': [0, 1, 2, 1.5, 1, 0.5],
        'y (mm)': [10, 12, 15, 13, 14, 16], # y (mm) for completeness, not directly used in HCD lofting if 'a','b' present
        'a': [10, 11, 14, 12, 13, 15],
        'b': [8, 9, 12, 10, 11, 13]
    }
    df_hcd_fold = pd.DataFrame(data_hcd_fold)
    step_bytes = generate_step(df_hcd_fold, hcd_enabled=True, fold=True)
    assert isinstance(step_bytes, bytes)
    assert len(step_bytes) > 0
    assert b"HEADER;" in step_bytes
    assert b"ENDSEC;" in step_bytes
    assert b"END-ISO-10303-21;" in step_bytes

# Test generate_dxf with minimal points (ezdxf spline requires at least 2 points for degree 1, 3 for degree 2, 4 for degree 3)
# The function uses default spline which is degree 3, so it needs at least 4 points.
# If less than 4 points, ezdxf might handle it by reducing degree or error.
# Let's check behavior with few points. Default spline in ezdxf adds control points if not enough.
@pytest.fixture
def sample_export_df_few_points():
    data = {'x (mm)': [0, 1, 2], 'y (mm)': [10, 12, 15]} # 3 points
    return pd.DataFrame(data)

def test_generate_dxf_few_points(sample_export_df_few_points):
    dxf_bytes = generate_dxf(sample_export_df_few_points)
    assert isinstance(dxf_bytes, bytes)
    assert len(dxf_bytes) > 0
    try:
        doc = ezdxf.read(BytesIO(dxf_bytes))
        assert doc is not None
        msp = doc.modelspace()
        splines = list(msp.query('SPLINE'))
        assert len(splines) > 0, "No SPLINE entity found in DXF modelspace with few points."
        # ezdxf's add_spline can create a spline even with fewer points by adjusting degree
        # or repeating points. Check that it doesn't raise an error.
    except ezdxf.DXFStructureError as e:
        pytest.fail(f"Failed to read DXF with few points: {e}")

def test_generate_dxf_two_points():
    data = {'x (mm)': [0, 1], 'y (mm)': [10, 12]} # 2 points
    df_two_points = pd.DataFrame(data)
    dxf_bytes = generate_dxf(df_two_points)
    assert isinstance(dxf_bytes, bytes)
    assert len(dxf_bytes) > 0
    try:
        doc = ezdxf.read(BytesIO(dxf_bytes))
        assert doc is not None
        msp = doc.modelspace()
        # With 2 points, it might create a LINE or a lower-degree SPLINE.
        # The current implementation of generate_dxf explicitly calls add_spline.
        # ezdxf's add_spline with 2 points will create a spline of degree 1.
        splines = list(msp.query('SPLINE'))
        assert len(splines) > 0
        assert splines[0].dxf.degree == 1
    except ezdxf.DXFStructureError as e:
        pytest.fail(f"Failed to read DXF with two points: {e}")
        
# Test generate_excel with an empty DataFrame
def test_generate_excel_empty_df():
    empty_df = pd.DataFrame(columns=['x (mm)', 'y (mm)'])
    excel_bytes = generate_excel(empty_df)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0 # xlsxwriter will still create a valid empty file
    
    df_read_back = pd.read_excel(BytesIO(excel_bytes))
    assert df_read_back.empty
    # Comparing columns might be tricky if original df had specific columns but no data
    # pd.testing.assert_frame_equal(df_read_back, empty_df, check_dtype=False)
    # For empty df, read_excel might not preserve columns unless they were written.
    # Let's check if the columns that were supposed to be there are indeed there.
    # If empty_df has columns defined, df_read_back should also have them.
    assert list(df_read_back.columns) == list(empty_df.columns)


# Test generate_dxf with an empty DataFrame
def test_generate_dxf_empty_df():
    empty_df = pd.DataFrame(columns=['x (mm)', 'y (mm)'])
    # Calling generate_dxf with an empty DataFrame will likely cause an error
    # when trying to access row['x (mm)'] or row['y (mm)'], or when msp.add_spline([])
    with pytest.raises(IndexError): # Or potentially another error depending on pandas/ezdxf behavior
        generate_dxf(empty_df)
        # If it doesn't raise an error (e.g., ezdxf handles empty points gracefully):
        # dxf_bytes = generate_dxf(empty_df)
        # assert isinstance(dxf_bytes, bytes)
        # doc = ezdxf.read(BytesIO(dxf_bytes))
        # msp = doc.modelspace()
        # assert len(list(msp.query('SPLINE'))) == 0


# Test generate_step with an empty DataFrame
def test_generate_step_empty_df():
    empty_df = pd.DataFrame(columns=['x (mm)', 'y (mm)'])
    # This will likely fail because the loop over df.iterrows() won't run,
    # and outer/inner workplanes won't be initialized or lofted correctly.
    # CadQuery might raise an error when trying to loft empty shapes.
    with pytest.raises(Exception): # cq.Workplane.loft might raise various errors for empty profiles
        generate_step(empty_df)
        # If it somehow produces a file (e.g. empty model):
        # step_bytes = generate_step(empty_df)
        # assert isinstance(step_bytes, bytes)
        # assert len(step_bytes) > 0 # Or it might be an empty/minimal STEP file

# Test generate_step with single point DataFrame (insufficient for loft)
def test_generate_step_single_point_df():
    single_point_df = pd.DataFrame({'x (mm)': [0], 'y (mm)': [10]})
    with pytest.raises(Exception): # CadQuery loft needs at least two profiles
        generate_step(single_point_df)

    single_point_hcd_df = pd.DataFrame({'x (mm)': [0], 'y (mm)': [10], 'a':[10], 'b':[8]})
    with pytest.raises(Exception):
        generate_step(single_point_hcd_df, hcd_enabled=True)

# Test generate_step with non-monotonic x values (not strictly decreasing for fold)
# This is to ensure fold logic handles various x sequences correctly.
def test_generate_step_fold_non_monotonic_x(sample_export_df):
    # sample_export_df has monotonic increasing x. Fold=True should still run.
    step_bytes = generate_step(sample_export_df, hcd_enabled=False, fold=True)
    assert isinstance(step_bytes, bytes)
    assert len(step_bytes) > 0

    # A more complex x series
    data_complex_x = {'x (mm)': [0, 1, 0.5, 1.5, 1], 'y (mm)': [10, 12, 11, 13, 12.5]}
    df_complex_x = pd.DataFrame(data_complex_x)
    step_bytes_complex = generate_step(df_complex_x, hcd_enabled=False, fold=True)
    assert isinstance(step_bytes_complex, bytes)
    assert len(step_bytes_complex) > 0

# Fixture for interpolate tests
@pytest.fixture
def sample_interpolate_df():
    data = {'x (mm)': [0, 10, 20, 30], 'y (mm)': [0, 5, 3, 10]}
    return pd.DataFrame(data)

# Tests for interpolate
def test_interpolate_basic(sample_interpolate_df):
    num_points_test = 5
    interpolated_df = interpolate(sample_interpolate_df, num_point=num_points_test)
    
    assert isinstance(interpolated_df, pd.DataFrame)
    assert len(interpolated_df) == num_points_test
    assert all(col in interpolated_df.columns for col in ['x (mm)', 'y (mm)'])
    
    expected_x = np.linspace(sample_interpolate_df['x (mm)'].min(), sample_interpolate_df['x (mm)'].max(), num_points_test)
    assert np.allclose(interpolated_df['x (mm)'], expected_x)
    
    assert np.isclose(interpolated_df['y (mm)'].iloc[0], sample_interpolate_df['y (mm)'].iloc[0])
    assert np.isclose(interpolated_df['y (mm)'].iloc[-1], sample_interpolate_df['y (mm)'].iloc[-1])

def test_interpolate_more_points(sample_interpolate_df):
    num_points_test = 100
    interpolated_df = interpolate(sample_interpolate_df, num_point=num_points_test)
    assert isinstance(interpolated_df, pd.DataFrame)
    assert len(interpolated_df) == num_points_test

def test_interpolate_few_points(sample_interpolate_df):
    # CubicSpline needs at least 2 points. If num_point=2, it should work.
    # np.linspace with num=2 will produce start and end points.
    num_points_test = 2
    interpolated_df = interpolate(sample_interpolate_df, num_point=num_points_test)
    assert isinstance(interpolated_df, pd.DataFrame)
    assert len(interpolated_df) == num_points_test
    
    assert np.isclose(interpolated_df['x (mm)'].iloc[0], sample_interpolate_df['x (mm)'].iloc[0])
    assert np.isclose(interpolated_df['y (mm)'].iloc[0], sample_interpolate_df['y (mm)'].iloc[0])
    assert np.isclose(interpolated_df['x (mm)'].iloc[-1], sample_interpolate_df['x (mm)'].iloc[-1])
    assert np.isclose(interpolated_df['y (mm)'].iloc[-1], sample_interpolate_df['y (mm)'].iloc[-1])

# Tests for create_2d_plot
def test_create_2d_plot(sample_interpolate_df):
    x_series = sample_interpolate_df['x (mm)']
    y_series = sample_interpolate_df['y (mm)']
    fig = create_2d_plot(x_series, y_series)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter)

def test_create_2d_plot_empty_data():
    # Test with empty Series
    fig = create_2d_plot(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # Plotly might still create a trace object for empty data
    assert isinstance(fig.data[0], go.Scatter)
    assert len(fig.data[0].x) == 0
    assert len(fig.data[0].y) == 0
    
    # Test with empty numpy arrays
    fig_np = create_2d_plot(np.array([]), np.array([]))
    assert isinstance(fig_np, go.Figure)
    assert len(fig_np.data) == 1
    assert isinstance(fig_np.data[0], go.Scatter)
    assert len(fig_np.data[0].x) == 0
    assert len(fig_np.data[0].y) == 0

# Tests for create_3d_plot
def test_create_3d_plot(sample_interpolate_df):
    x_series = sample_interpolate_df['x (mm)']
    y_series = sample_interpolate_df['y (mm)']
    fig = create_3d_plot(x_series, y_series)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter3d)

def test_create_3d_plot_empty_data():
    # Test with empty Series
    fig = create_3d_plot(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1 # Plotly might still create a trace object for empty data
    assert isinstance(fig.data[0], go.Scatter3d)
    # The internal loop for X,Y,Z will not run if x is empty.
    # So fig.data[0].x (etc.) should be None or empty depending on Plotly's behavior.
    # Based on how X,Y,Z are constructed, they will be empty lists.
    assert len(fig.data[0].x) == 0 
    assert len(fig.data[0].y) == 0
    assert len(fig.data[0].z) == 0

    # Test with empty numpy arrays
    fig_np = create_3d_plot(np.array([]), np.array([]))
    assert isinstance(fig_np, go.Figure)
    assert len(fig_np.data) == 1
    assert isinstance(fig_np.data[0], go.Scatter3d)
    assert len(fig_np.data[0].x) == 0
    assert len(fig_np.data[0].y) == 0
    assert len(fig_np.data[0].z) == 0

# Test interpolate with dataframe having non-default index
def test_interpolate_non_default_index(sample_interpolate_df):
    df_non_default_idx = sample_interpolate_df.copy()
    df_non_default_idx.index = [10, 20, 30, 40] # Arbitrary index
    num_points_test = 5
    interpolated_df = interpolate(df_non_default_idx, num_point=num_points_test)
    
    assert isinstance(interpolated_df, pd.DataFrame)
    assert len(interpolated_df) == num_points_test
    # Check that the index of the result is a default RangeIndex
    assert isinstance(interpolated_df.index, pd.RangeIndex)
    assert interpolated_df.index.start == 0
    assert interpolated_df.index.stop == num_points_test
    assert interpolated_df.index.step == 1
    
    expected_x = np.linspace(df_non_default_idx['x (mm)'].min(), df_non_default_idx['x (mm)'].max(), num_points_test)
    assert np.allclose(interpolated_df['x (mm)'], expected_x)
    assert np.isclose(interpolated_df['y (mm)'].iloc[0], df_non_default_idx['y (mm)'].iloc[0])
    assert np.isclose(interpolated_df['y (mm)'].iloc[-1], df_non_default_idx['y (mm)'].iloc[-1])

# Test interpolate with only two data points in input (minimum for CubicSpline)
def test_interpolate_two_input_points():
    data = {'x (mm)': [0, 10], 'y (mm)': [0, 5]}
    df_two_points = pd.DataFrame(data)
    num_points_test = 5
    interpolated_df = interpolate(df_two_points, num_point=num_points_test)
    
    assert isinstance(interpolated_df, pd.DataFrame)
    assert len(interpolated_df) == num_points_test
    expected_x = np.linspace(0, 10, num_points_test)
    assert np.allclose(interpolated_df['x (mm)'], expected_x)
    # For linear interpolation between 2 points, y should also be on the line
    expected_y = np.linspace(0, 5, num_points_test) # CubicSpline with 2 points should be linear
    assert np.allclose(interpolated_df['y (mm)'], expected_y)

# Test create_2d_plot with numpy array inputs
def test_create_2d_plot_numpy_input(sample_interpolate_df):
    x_np = sample_interpolate_df['x (mm)'].to_numpy()
    y_np = sample_interpolate_df['y (mm)'].to_numpy()
    fig = create_2d_plot(x_np, y_np)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter)
    assert np.array_equal(fig.data[0].x, x_np)
    assert np.array_equal(fig.data[0].y, y_np)

# Test create_3d_plot with numpy array inputs
def test_create_3d_plot_numpy_input(sample_interpolate_df):
    x_np = sample_interpolate_df['x (mm)'].to_numpy()
    y_np = sample_interpolate_df['y (mm)'].to_numpy()
    fig = create_3d_plot(x_np, y_np)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatter3d)
    # Check if the generated points for 3D plot are based on input,
    # this is a bit more involved as the points are transformed.
    # For now, just checking type and that it runs is primary.
    # Number of points in Scatter3d should be len(x_np) * 50 (theta resolution)
    assert len(fig.data[0].x) == len(x_np) * 50

# Test interpolate with a single data point in input DataFrame
# CubicSpline requires at least 2 points. This should raise an error.
def test_interpolate_single_input_point():
    data = {'x (mm)': [0], 'y (mm)': [0]}
    df_single_point = pd.DataFrame(data)
    with pytest.raises(ValueError) as excinfo: # From CubicSpline: x must have at least 2 elements
        interpolate(df_single_point, num_point=5)
    assert "at least 2" in str(excinfo.value).lower() or "k must be integer" in str(excinfo.value).lower() # SciPy error messages vary
    
# Test interpolate where num_point is less than 2
def test_interpolate_num_point_less_than_two(sample_interpolate_df):
    with pytest.raises(ValueError) as excinfo: # From np.linspace: num must be non-negative
        interpolate(sample_interpolate_df, num_point=1)
    assert "num must be non-negative" in str(excinfo.value) or "must be at least 2" in str(excinfo.value) # Linspace error or Spline error

    with pytest.raises(ValueError):
        interpolate(sample_interpolate_df, num_point=0)
    
    with pytest.raises(TypeError): # num_point should be an integer
        interpolate(sample_interpolate_df, num_point=3.5)
