import csv
import os
from enum import Enum
from pathlib import Path


class DATA_SCHEME(Enum):
    GAME = "game"
    TRAINING_MODE = "training_model"    # Either mnih2013, mnih2015, or dqn_with_huber_loss_and_adam
    REWARD_HISTORY = "reward_history"
    LOSS_HISTORY = "loss_history"
    MODEL_PATH = "model_path"

    @staticmethod
    def get_col_names() -> tuple[str]:
        return tuple(str(col.value) for col in DATA_SCHEME)


def append_to_csv(csv_path, input_row):
    """
    Append a new row to an existing csv file and close the file afterwards. Creates the csv if it doesn't exist yet.

    :param csv_path: Path to the csv file.
    :param input_row:  New row to add at the end of the file. Must follow the DATA_SCHEME.
    """
    col_names = DATA_SCHEME.get_col_names()
    assert len(input_row) == len(col_names), f"Wrong amount of elements in input row ({len(input_row)} instead" \
                                             f" of {len(col_names)}). input_row = {input_row}"

    if not os.path.exists(csv_path):
        Path(os.path.dirname(csv_path)).mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f_object:
            writer = csv.writer(f_object)
            writer.writerow(col_names)

    with open(csv_path, "a") as f_object:
        writer = csv.writer(f_object)
        writer.writerow(input_row)
