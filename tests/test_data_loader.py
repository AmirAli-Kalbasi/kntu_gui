import numpy as np
from scipy.io import savemat
from models import load_test_data


def _make_mat(path):
    data = np.ones((1, 7))
    savemat(path, {"dataset": data})


def test_load_test_data_with_files_in_root(tmp_path):
    mat_path = tmp_path / "sample.mat"
    _make_mat(mat_path)

    df = load_test_data(tmp_path)
    assert not df.empty
    assert df["folder"].iloc[0] == tmp_path.name
    assert "label" not in df.columns or df["label"].isna().all()


def test_load_test_data_with_labelled_folders(tmp_path):
    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    fault_dir = tmp_path / "fault"
    fault_dir.mkdir()
    _make_mat(normal_dir / "n.mat")
    _make_mat(fault_dir / "f.mat")

    df = load_test_data(tmp_path)
    assert set(df["label"]) == {"normal", "fault"}

