import os
import logging
import numpy as np
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)


DEFAULT_COLUMNS = [
    "sample_num",
    "Acc_X",
    "Acc_Y",
    "Acc_Z",
    "Gyro_X",
    "Gyro_Y",
    "Gyro_Z",
]


def _dataframe_from_dataset(data_array):
    """Create a DataFrame from a loaded ``dataset`` array.

    The legacy format stores only numeric values. Some newer files ship the first
    row as column headers; when detected those are used instead of the default
    ``Acc``/``Gyro`` labels.
    """

    columns = DEFAULT_COLUMNS
    values = data_array

    if data_array.size:
        first_row = np.asarray(data_array[0]).ravel()
        if any(isinstance(item, str) for item in first_row):
            columns = [str(item) for item in first_row]
            values = data_array[1:]

    return pd.DataFrame(values, columns=columns)


def load_and_label_data(base_path, verbose=True):
    """Load ``.mat`` files from subfolders containing normal and fault data.
    
        By default the function looks for ``Positive``/``Negative`` folders but it
        also accepts the more explicit ``fault``/``normal`` naming. Any of these
        directories will be processed if present within ``base_path``. Files found
        under a fault folder are labeled as faults and files found under a normal
        folder are labeled as normal.
    
        Parameters
        ----------
        base_path : str
            Path containing the ``Positive`` and ``Negative`` folders.
    
        Returns
        -------
        tuple[pandas.DataFrame, dict]
            Combined data from all files with a ``label`` column and a dictionary
            with counts of ``fault`` and ``normal`` files.
    
        Notes
        -----
        Set ``verbose`` to ``True`` to log a summary of loaded files using the
        standard ``logging`` module.
        """
    data_frames = []
    counts = {"fault": 0, "normal": 0}

    for subfolder in os.listdir(base_path):
        folder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(folder_path):
            continue

        name = subfolder.lower()
        if name in {"positive", "fault"}:
            label = "fault"
        elif name in {"negative", "normal"}:
            label = "normal"
        else:
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".mat"):
                file_path = os.path.join(folder_path, file)
                mat_data = scipy.io.loadmat(file_path)

                data_array = mat_data["dataset"]
                df = _dataframe_from_dataset(data_array)
                df["label"] = label
                df["source_file"] = file
                data_frames.append(df)
                counts[label] += 1

    if data_frames:
        combined = pd.concat(data_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    if verbose:
        logger.info(
            "Loaded %s normal files and %s fault files.",
            counts["normal"],
            counts["fault"],
        )

    return combined, counts


def load_test_data(base_path):
    """Load ``.mat`` files for testing.

    Files can either be placed directly in ``base_path`` or inside
    subfolders.  When folders named ``fault``/``normal`` (or their
    ``Positive``/``Negative`` variants) are present, the returned data will
    include a ``label`` column allowing a confusion matrix to be
    computed.
    """

    data_frames = []

    for entry in os.listdir(base_path):
        path = os.path.join(base_path, entry)

        # Handle files placed directly in the base path
        if os.path.isfile(path) and entry.endswith(".mat"):
            mat_data = scipy.io.loadmat(path)
            data_array = mat_data["dataset"]
            df = _dataframe_from_dataset(data_array)
            df["folder"] = os.path.basename(base_path)
            df["source_file"] = entry
            data_frames.append(df)
            continue

        # Skip non-directories after handling direct files
        if not os.path.isdir(path):
            continue

        name = entry.lower()
        label = None
        if name in {"negative", "fault"}:
            label = "fault"
        elif name in {"positive", "normal"}:
            label = "normal"

        for file in os.listdir(path):
            if file.endswith(".mat"):
                file_path = os.path.join(path, file)
                mat_data = scipy.io.loadmat(file_path)

                data_array = mat_data["dataset"]
                df = _dataframe_from_dataset(data_array)
                df["folder"] = entry
                df["source_file"] = file
                if label is not None:
                    df["label"] = label
                data_frames.append(df)

    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    return pd.DataFrame()
