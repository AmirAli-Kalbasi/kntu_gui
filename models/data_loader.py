import os
import logging
import pandas as pd
import scipy.io

logger = logging.getLogger(__name__)


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
                df = pd.DataFrame(
                    data_array,
                    columns=[
                        "sample_num",
                        "Acc_X",
                        "Acc_Y",
                        "Acc_Z",
                        "Gyro_X",
                        "Gyro_Y",
                        "Gyro_Z",
                    ],
                )
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
