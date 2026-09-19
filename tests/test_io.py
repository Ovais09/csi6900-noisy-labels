import pandas as pd

from src.utils.io import load_dataframe, save_dataframe_atomic


def test_csv_helpers_round_trip_floating_point_values_exactly(tmp_path):
    path = tmp_path / "floats.csv"
    expected = pd.DataFrame(
        {
            "value": [
                0.9122807017543859,
                0.9473684210526315,
                0.8235294117647058,
            ]
        }
    )

    save_dataframe_atomic(expected, path)
    actual = load_dataframe(path)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
