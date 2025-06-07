import os
import pandas as pd
import scipy.io


def load_and_label_data(base_path, verbose=True):
    """Load ``.mat`` files from Positive and Negative subfolders.

    The function expects two directories inside ``base_path`` named
    ``Positive`` and ``Negative``. Files found under ``Positive`` are labeled
    as faults and files found under ``Negative`` are labeled as normal.

    Parameters
    ----------
    base_path : str
        Path containing the ``Positive`` and ``Negative`` folders.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        Combined data from all files with a ``label`` column and a dictionary
        with counts of ``fault`` and ``normal`` files.
    """
    data_frames = []
    counts = {"fault": 0, "normal": 0}

    for subfolder in ["Positive", "Negative"]:
        folder_path = os.path.join(base_path, subfolder)
        if not os.path.isdir(folder_path):
            continue

        label = "fault" if subfolder == "Positive" else "normal"
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
        print(
            f"Loaded {counts['normal']} normal files "
            f"and {counts['fault']} fault files."
        )

    return combined, counts
