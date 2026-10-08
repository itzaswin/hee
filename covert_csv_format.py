import os
import pickle
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


class format_1:

    def __init__(self, fpath, spath):
        self.fpath = fpath
        self.spath = spath

    def convert_folder_to_csv(self):
        for dirpath, _, filenames in os.walk(self.fpath):
            rel_path = os.path.relpath(
                dirpath,
                self.fpath
            )

            out_dir = os.path.join(
                self.spath,
                rel_path
            )

            os.makedirs(
                out_dir,
                exist_ok=True
            )

            for file in filenames:
                file_path = os.path.join(
                    dirpath,
                    file
                )

                ext = os.path.splitext(
                    file
                )[1].lower()

                base_name = os.path.splitext(
                    file
                )[0]

                if ext == ".csv":
                    self.process_csv(
                        file_path,
                        out_dir,
                        file
                    )

                elif ext == ".dat":
                    self.process_dat(
                        file_path,
                        out_dir,
                        base_name
                    )

                elif ext == ".xml":
                    self.process_xml(
                        file_path,
                        out_dir,
                        base_name
                    )

                elif ext == ".txt":
                    self.process_txt(
                        file_path,
                        out_dir,
                        base_name
                    )

                elif ext in [".xlsx", ".xls"]:
                    self.process_excel(
                        file_path,
                        out_dir,
                        base_name
                    )

    def clean_value(self, value):
        if pd.isna(value):
            return ""

        value = str(value).strip()

        value = value.replace('"', "")
        value = value.replace("(", "")
        value = value.replace(")", "")
        value = value.replace("\\", "")

        return value.strip()

    def process_csv(
        self,
        file_path,
        out_dir,
        file_name
    ):
        try:
            out_path = os.path.join(
                out_dir,
                file_name
            )

            shutil.copy2(
                file_path,
                out_path
            )

            print(
                "Copied CSV file:",
                file_path
            )

        except Exception as e:
            print(
                "Error in CSV file:",
                file_path,
                "->",
                e
            )

    def process_dat(
        self,
        file_path,
        out_dir,
        base_name
    ):
        try:
            with open(
                file_path,
                "rb"
            ) as f:
                loaded_data = pickle.load(f)

            if isinstance(
                loaded_data,
                dict
            ):
                for key, val in loaded_data.items():

                    if isinstance(
                        val,
                        (np.ndarray, list)
                    ):
                        df = pd.DataFrame(val)

                        df = self.clean_dataframe(
                            df
                        )

                        out_path = os.path.join(
                            out_dir,
                            f"{base_name}_{key}.csv"
                        )

                        self.write_csv(
                            df,
                            out_path
                        )

            elif isinstance(
                loaded_data,
                (np.ndarray, list)
            ):
                df = pd.DataFrame(
                    loaded_data
                )

                df = self.clean_dataframe(
                    df
                )

                out_path = os.path.join(
                    out_dir,
                    f"{base_name}.csv"
                )

                self.write_csv(
                    df,
                    out_path
                )

            print(
                "Converted DAT file:",
                file_path
            )

        except Exception as e:
            print(
                "Error in DAT file:",
                file_path,
                "->",
                e
            )

    def process_xml(
        self,
        file_path,
        out_dir,
        base_name
    ):
        try:
            tree = ET.parse(
                file_path
            )

            root = tree.getroot()

            rows = []

            for eye_movement in root.findall(
                ".//EyeMovement"
            ):
                row = {}

                def extract_attributes(
                    element,
                    path
                ):
                    if path:
                        current_path = (
                            f"{path}_{element.tag}"
                        )
                    else:
                        current_path = (
                            element.tag
                        )

                    for attr_name, attr_value in (
                        element.attrib.items()
                    ):
                        if attr_name.lower() in [
                            "valid",
                            "validity"
                        ]:
                            continue

                        value = self.clean_value(
                            attr_value
                        )

                        if value:
                            column_name = (
                                f"{current_path}_"
                                f"{attr_name}"
                            )

                            row[column_name] = value

                    for child in element:
                        extract_attributes(
                            child,
                            current_path
                        )

                for child in eye_movement:
                    extract_attributes(
                        child,
                        ""
                    )

                if row:
                    rows.append(row)

            if not rows:
                print(
                    "No feature data found:",
                    file_path
                )
                return

            df = pd.DataFrame(
                rows
            )

            df = self.clean_dataframe(
                df
            )

            out_path = os.path.join(
                out_dir,
                f"{base_name}.csv"
            )

            self.write_csv(
                df,
                out_path
            )

            print(
                "Converted XML file:",
                file_path,
                "-> Rows:",
                len(df),
                "Columns:",
                len(df.columns)
            )

        except Exception as e:
            print(
                "Error in XML file:",
                file_path,
                "->",
                e
            )

    def process_txt(
        self,
        file_path,
        out_dir,
        base_name
    ):
        try:
            rows = []

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                for line in f:
                    line = self.clean_value(
                        line
                    )

                    if line:
                        rows.append(
                            [line]
                        )

            if not rows:
                print(
                    "No data found in TXT file:",
                    file_path
                )
                return

            df = pd.DataFrame(
                rows
            )

            df = self.clean_dataframe(
                df
            )

            out_path = os.path.join(
                out_dir,
                f"{base_name}.csv"
            )

            self.write_csv(
                df,
                out_path
            )

            print(
                "Converted TXT file:",
                file_path,
                "-> Rows:",
                len(df)
            )

        except Exception as e:
            print(
                "Error in TXT file:",
                file_path,
                "->",
                e
            )

    def process_excel(
        self,
        file_path,
        out_dir,
        base_name
    ):
        try:
            df = pd.read_excel(
                file_path
            )

            df = self.clean_dataframe(
                df
            )

            out_path = os.path.join(
                out_dir,
                f"{base_name}.csv"
            )

            self.write_csv(
                df,
                out_path
            )

            print(
                "Converted Excel file:",
                file_path
            )

        except Exception as e:
            print(
                "Error in Excel file:",
                file_path,
                "->",
                e
            )

    def clean_dataframe(
        self,
        df
    ):
        df = df.fillna("")

        df.columns = [
            self.clean_value(column)
            for column in df.columns
        ]

        for column in df.columns:
            df[column] = (
                df[column]
                .apply(self.clean_value)
            )

        return df

    def write_csv(
        self,
        df,
        out_path
    ):
        with open(
            out_path,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            columns = [
                self.clean_value(column)
                for column in df.columns
            ]

            f.write(
                ",".join(columns)
            )

            f.write("\n")

            for row_data in df.itertuples(
                index=False,
                name=None
            ):
                values = []

                for value in row_data:
                    value = self.clean_value(
                        value
                    )

                    values.append(
                        value
                    )

                f.write(
                    ",".join(values)
                )

                f.write("\n")


import os
import re
import pickle
import shutil
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class format_2:

    def __init__(self, fpath, spath, head_fpath):

        self.fpath = os.path.abspath(fpath)
        self.spath = os.path.abspath(spath)
        self.head_fpath = os.path.abspath(head_fpath)

        self.ecg_gsr_fpath = os.path.join(
            self.fpath,
            "01_ECG_GSR"
        )

        self.eye_fpath = os.path.join(
            self.fpath,
            "02_EYE_Movement"
        )

        self.questionnaire_fpath = os.path.join(
            self.fpath,
            "03_questionary"
        )

        self.ecg_folder = os.path.join(
            self.spath,
            "01_ECG"
        )

        self.gsr_folder = os.path.join(
            self.spath,
            "02_GSR"
        )

        self.eye_folder = os.path.join(
            self.spath,
            "03_EYE_Movement"
        )

        self.head_folder = os.path.join(
            self.spath,
            "04_HEAD_Movement"
        )

        self.questionnaire_folder = os.path.join(
            self.spath,
            "05_Questionnaires"
        )

        self.target_folder = os.path.join(
            self.spath,
            "06_Target"
        )

        self.max_videos = 12

        self.participants = [
            f"P{number}"
            for number in range(300, 335)
        ]

        self.videos = [
            f"V{number}"
            for number in range(1, 13)
        ]

        self.create_output_folders()

    def create_output_folders(self):

        folders = [
            self.spath,
            self.ecg_folder,
            self.gsr_folder,
            self.eye_folder,
            self.head_folder,
            self.questionnaire_folder,
            self.target_folder
        ]

        for folder in folders:

            os.makedirs(
                folder,
                exist_ok=True
            )

    def clean_output(self):

        if os.path.isdir(self.spath):

            folders = [
                "01_ECG",
                "02_GSR",
                "03_EYE_Movement",
                "04_HEAD_Movement",
                "05_Questionnaires",
                "06_Target",
                "04_Questionnaires",
                "05_Target"
            ]

            for folder_name in folders:

                folder_path = os.path.join(
                    self.spath,
                    folder_name
                )

                if os.path.isdir(folder_path):

                    shutil.rmtree(
                        folder_path
                    )

        self.create_output_folders()

    def is_temporary_file(self, filename):

        return (
            filename.startswith("~$")
            or filename.startswith(".~")
            or filename.endswith("~")
        )

    def get_participant_id(self, value):

        text = os.path.basename(
            str(value)
        )

        match = re.search(
            r"(?<!\d)(P3(?:00|0[1-9]|1[0-9]|2[0-9]|3[0-4]))(?!\d)",
            text,
            re.IGNORECASE
        )

        if match is not None:

            participant_id = match.group(
                1
            ).upper()

            number = int(
                participant_id[1:]
            )

            if 300 <= number <= 334:

                return participant_id

        match = re.search(
            r"(?<!\d)(10[1-9]|1[1-2][0-9]|13[0-5])(?!\d)",
            text
        )

        if match is not None:

            source_id = int(
                match.group(1)
            )

            if 101 <= source_id <= 135:

                return f"P{source_id + 199}"

        return None

    def get_video_id(self, value):

        if value is None:

            return None

        try:

            number = int(
                float(value)
            )

            if 1 <= number <= 12:

                return f"V{number}"

        except Exception:

            pass

        text = str(
            value
        ).strip()

        match = re.search(
            r"V\s*([0-9]+)",
            text,
            re.IGNORECASE
        )

        if match is not None:

            number = int(
                match.group(1)
            )

            if 1 <= number <= 12:

                return f"V{number}"

        return None

    def load_dat(self, path):

        try:

            with open(
                path,
                "rb"
            ) as file:

                return pickle.load(
                    file,
                    encoding="latin1"
                )

        except Exception:

            pass

        try:

            return pd.read_pickle(
                path
            )

        except Exception:

            pass

        try:

            return np.load(
                path,
                allow_pickle=True
            )

        except Exception:

            pass

        return None

    def normalize_dataframe(self, dataframe):

        if not isinstance(
            dataframe,
            pd.DataFrame
        ):

            return None

        dataframe = dataframe.copy()

        dataframe.reset_index(
            drop=True,
            inplace=True
        )

        if isinstance(
            dataframe.columns,
            pd.MultiIndex
        ):

            dataframe.columns = [
                "_".join(
                    [
                        str(value)
                        for value in column
                        if str(value) != "nan"
                    ]
                )
                for column in dataframe.columns
            ]

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        return dataframe

    def add_metadata(
        self,
        dataframe,
        participant_id,
        video_id
    ):

        dataframe = self.normalize_dataframe(
            dataframe
        )

        if dataframe is None:

            return None

        remove_columns = []

        for column in dataframe.columns:

            column_name = str(
                column
            ).lower()

            if column_name in [
                "participant_id",
                "video_id"
            ]:

                remove_columns.append(
                    column
                )

        if remove_columns:

            dataframe.drop(
                columns=remove_columns,
                inplace=True
            )

        dataframe.insert(
            0,
            "Participant_ID",
            participant_id
        )

        dataframe.insert(
            1,
            "Video_ID",
            video_id
        )

        return dataframe

    def save_dataframe(
        self,
        dataframe,
        folder,
        filename
    ):

        dataframe = self.normalize_dataframe(
            dataframe
        )

        if dataframe is None:

            return

        if dataframe.empty:

            return

        os.makedirs(
            folder,
            exist_ok=True
        )

        output_path = os.path.join(
            folder,
            filename
        )

        dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig"
        )

    def find_dataframes(
        self,
        obj,
        result=None,
        depth=0,
        max_depth=12
    ):

        if result is None:

            result = []

        if obj is None:

            return result

        if depth > max_depth:

            return result

        if isinstance(
            obj,
            pd.DataFrame
        ):

            result.append(
                obj
            )

            return result

        if isinstance(
            obj,
            dict
        ):

            for value in obj.values():

                self.find_dataframes(
                    value,
                    result,
                    depth + 1,
                    max_depth
                )

        elif isinstance(
            obj,
            (
                list,
                tuple
            )
        ):

            for value in obj:

                self.find_dataframes(
                    value,
                    result,
                    depth + 1,
                    max_depth
                )

        return result

    def get_recordings(self, data):

        if isinstance(
            data,
            dict
        ):

            if "Data" in data:

                return data["Data"]

            if "data" in data:

                return data["data"]

        return data

    def extract_channel_dataframe(
        self,
        channel,
        feature_name,
        sampling_rate=None
    ):

        if isinstance(
            channel,
            pd.DataFrame
        ):

            return self.normalize_dataframe(
                channel
            )

        if isinstance(
            channel,
            pd.Series
        ):

            values = channel.to_numpy()

        else:

            try:

                values = np.asarray(
                    channel
                )

            except Exception:

                return None

        if values.ndim == 0:

            return None

        if values.ndim == 1:

            dataframe = pd.DataFrame({
                feature_name: values
            })

        elif values.ndim == 2:

            if values.shape[0] == 1:

                dataframe = pd.DataFrame({
                    feature_name: values[0]
                })

            elif values.shape[1] == 1:

                dataframe = pd.DataFrame({
                    feature_name: values[:, 0]
                })

            elif values.shape[0] < values.shape[1]:

                dataframe = pd.DataFrame(
                    values.T
                )

                dataframe.columns = [
                    f"{feature_name}_{index + 1:02d}"
                    for index in range(
                        dataframe.shape[1]
                    )
                ]

            else:

                dataframe = pd.DataFrame(
                    values
                )

                dataframe.columns = [
                    f"{feature_name}_{index + 1:02d}"
                    for index in range(
                        dataframe.shape[1]
                    )
                ]

        else:

            return None

        if (
            feature_name == "ECG"
            and "time" not in dataframe.columns
            and sampling_rate is not None
        ):

            dataframe.insert(
                0,
                "time",
                np.arange(
                    len(dataframe)
                ) / sampling_rate
            )

        return dataframe

    def save_ecg_gsr_array_recording(
        self,
        recording,
        participant_id,
        video_index
    ):

        if isinstance(
            recording,
            np.ndarray
        ):

            if recording.ndim != 2:

                return

            if recording.shape[0] >= 2:

                gsr_signal = recording[0]
                ecg_signal = recording[1]

            elif recording.shape[1] >= 2:

                gsr_signal = recording[:, 0]
                ecg_signal = recording[:, 1]

            else:

                return

        elif isinstance(
            recording,
            (
                list,
                tuple
            )
        ):

            if len(recording) < 2:

                return

            gsr_signal = recording[0]
            ecg_signal = recording[1]

        else:

            return

        video_id = f"V{video_index}"

        gsr_dataframe = self.extract_channel_dataframe(
            gsr_signal,
            "GSR",
            2000
        )

        ecg_dataframe = self.extract_channel_dataframe(
            ecg_signal,
            "ECG",
            1000
        )

        if gsr_dataframe is not None:

            gsr_dataframe = self.add_metadata(
                gsr_dataframe,
                participant_id,
                video_id
            )

            self.save_dataframe(
                gsr_dataframe,
                self.gsr_folder,
                f"gsr_{participant_id}_{video_id}.csv"
            )

        if ecg_dataframe is not None:

            ecg_dataframe = self.add_metadata(
                ecg_dataframe,
                participant_id,
                video_id
            )

            self.save_dataframe(
                ecg_dataframe,
                self.ecg_folder,
                f"ecg_{participant_id}_{video_id}.csv"
            )

    def extract_ecg_gsr(
        self,
        path,
        participant_id
    ):

        data = self.load_dat(
            path
        )

        if data is None:

            return

        dataframes = self.find_dataframes(
            data
        )

        ecg_frames = []
        gsr_frames = []

        for dataframe in dataframes:

            ecg_columns = [
                column
                for column in dataframe.columns
                if "ecg" in str(column).lower()
            ]

            gsr_columns = [
                column
                for column in dataframe.columns
                if (
                    "gsr" in str(column).lower()
                    or "skin_conductance" in str(column).lower()
                    or "skin_resistance" in str(column).lower()
                    or "shimmer" in str(column).lower()
                )
            ]

            if ecg_columns:

                selected = []

                for column in dataframe.columns:

                    text = str(
                        column
                    ).lower()

                    if (
                        "ecg" in text
                        or "time" in text
                        or "timestamp" in text
                    ):

                        selected.append(
                            column
                        )

                if selected:

                    ecg_frames.append(
                        dataframe[
                            selected
                        ].copy()
                    )

            if gsr_columns:

                selected = []

                for column in dataframe.columns:

                    text = str(
                        column
                    ).lower()

                    if (
                        "gsr" in text
                        or "skin_conductance" in text
                        or "skin_resistance" in text
                        or "shimmer" in text
                        or "time" in text
                        or "timestamp" in text
                    ):

                        selected.append(
                            column
                        )

                if selected:

                    gsr_frames.append(
                        dataframe[
                            selected
                        ].copy()
                    )

        if ecg_frames:

            for index, dataframe in enumerate(
                ecg_frames[:self.max_videos],
                start=1
            ):

                dataframe = self.add_metadata(
                    dataframe,
                    participant_id,
                    f"V{index}"
                )

                self.save_dataframe(
                    dataframe,
                    self.ecg_folder,
                    f"ecg_{participant_id}_V{index}.csv"
                )

        if gsr_frames:

            for index, dataframe in enumerate(
                gsr_frames[:self.max_videos],
                start=1
            ):

                dataframe = self.add_metadata(
                    dataframe,
                    participant_id,
                    f"V{index}"
                )

                self.save_dataframe(
                    dataframe,
                    self.gsr_folder,
                    f"gsr_{participant_id}_V{index}.csv"
                )

        if ecg_frames or gsr_frames:

            return

        recordings = self.get_recordings(
            data
        )

        if isinstance(
            recordings,
            np.ndarray
        ):

            if recordings.ndim == 3:

                count = min(
                    recordings.shape[0],
                    self.max_videos
                )

                for index in range(
                    count
                ):

                    self.save_ecg_gsr_array_recording(
                        recordings[index],
                        participant_id,
                        index + 1
                    )

                return

        if isinstance(
            recordings,
            (
                list,
                tuple
            )
        ):

            count = min(
                len(recordings),
                self.max_videos
            )

            for index in range(
                count
            ):

                self.save_ecg_gsr_array_recording(
                    recordings[index],
                    participant_id,
                    index + 1
                )

    def extract_eye_array_recording(
        self,
        recording,
        participant_id,
        video_index
    ):

        try:

            array = np.asarray(
                recording
            )

        except Exception:

            return

        if array.ndim != 2:

            return

        if array.shape[0] <= array.shape[1]:

            values = array.T

        else:

            values = array

        if values.shape[1] == 0:

            return

        columns = []

        for index in range(
            values.shape[1]
        ):

            if index == 0:

                columns.append(
                    "Timestamp"
                )

            else:

                columns.append(
                    f"Eye_Feature_{index:02d}"
                )

        dataframe = pd.DataFrame(
            values,
            columns=columns
        )

        dataframe = self.add_metadata(
            dataframe,
            participant_id,
            f"V{video_index}"
        )

        self.save_dataframe(
            dataframe,
            self.eye_folder,
            f"eye_{participant_id}_V{video_index}.csv"
        )

    def extract_eye(
        self,
        path,
        participant_id
    ):

        data = self.load_dat(
            path
        )

        if data is None:

            return

        recordings = self.get_recordings(
            data
        )

        if isinstance(
            recordings,
            np.ndarray
        ):

            if recordings.ndim == 3:

                count = min(
                    recordings.shape[0],
                    self.max_videos
                )

                for index in range(
                    count
                ):

                    self.extract_eye_array_recording(
                        recordings[index],
                        participant_id,
                        index + 1
                    )

            return

        if isinstance(
            recordings,
            (
                list,
                tuple
            )
        ):

            count = min(
                len(recordings),
                self.max_videos
            )

            for index in range(
                count
            ):

                recording = recordings[index]

                if isinstance(
                    recording,
                    pd.DataFrame
                ):

                    dataframe = self.add_metadata(
                        recording,
                        participant_id,
                        f"V{index + 1}"
                    )

                    self.save_dataframe(
                        dataframe,
                        self.eye_folder,
                        f"eye_{participant_id}_V{index + 1}.csv"
                    )

                else:

                    self.extract_eye_array_recording(
                        recording,
                        participant_id,
                        index + 1
                    )

    def process_dat_files(self):

        if os.path.isdir(
            self.ecg_gsr_fpath
        ):

            for root, directories, files in os.walk(
                self.ecg_gsr_fpath
            ):

                for filename in files:

                    if self.is_temporary_file(
                        filename
                    ):

                        continue

                    if not filename.lower().endswith(
                        ".dat"
                    ):

                        continue

                    participant_id = self.get_participant_id(
                        filename
                    )

                    if participant_id is None:

                        continue

                    path = os.path.join(
                        root,
                        filename
                    )

                    self.extract_ecg_gsr(
                        path,
                        participant_id
                    )

        if os.path.isdir(
            self.eye_fpath
        ):

            for root, directories, files in os.walk(
                self.eye_fpath
            ):

                for filename in files:

                    if self.is_temporary_file(
                        filename
                    ):

                        continue

                    if not filename.lower().endswith(
                        ".dat"
                    ):

                        continue

                    participant_id = self.get_participant_id(
                        filename
                    )

                    if participant_id is None:

                        continue

                    path = os.path.join(
                        root,
                        filename
                    )

                    self.extract_eye(
                        path,
                        participant_id
                    )

    def read_table(self, path):

        filename = os.path.basename(
            path
        )

        if self.is_temporary_file(
            filename
        ):

            return None

        extension = os.path.splitext(
            path
        )[1].lower()

        try:

            if extension == ".csv":

                return pd.read_csv(
                    path,
                    low_memory=False
                )

            if extension in [
                ".xlsx",
                ".xls"
            ]:

                return pd.read_excel(
                    path
                )

            if extension == ".tsv":

                return pd.read_csv(
                    path,
                    sep="\t",
                    low_memory=False
                )

            if extension == ".txt":

                return pd.read_csv(
                    path,
                    sep=None,
                    engine="python",
                    low_memory=False
                )

        except Exception:

            pass

        return None

    def get_head_columns(
        self,
        dataframe
    ):

        head_columns = []

        for column in dataframe.columns:

            text = str(
                column
            ).lower()

            if any(
                key in text
                for key in [
                    "hmdrx",
                    "hmdry",
                    "hmdrz",
                    "head",
                    "rotation",
                    "orientation",
                    "pose_position",
                    "pose_rotation",
                    "yaw",
                    "pitch",
                    "roll",
                    "timestamp",
                    "timestamp",
                    "time"
                ]
            ):

                head_columns.append(
                    column
                )

        if not head_columns:

            head_columns = list(
                dataframe.columns
            )

        return head_columns

    def get_head_source_files(self):

        source_files = []

        if not os.path.isdir(
            self.head_fpath
        ):

            return source_files

        for root, directories, files in os.walk(
            self.head_fpath
        ):

            for filename in files:

                if self.is_temporary_file(
                    filename
                ):

                    continue

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension not in [
                    ".csv",
                    ".txt",
                    ".tsv",
                    ".xlsx",
                    ".xls"
                ]:

                    continue

                source_files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

        return source_files

    def process_head_files(self):

        source_files = self.get_head_source_files()

        if not source_files:

            return

        valid_files = []

        for source_path in source_files:

            dataframe = self.read_table(
                source_path
            )

            if dataframe is None:

                continue

            dataframe = self.normalize_dataframe(
                dataframe
            )

            if dataframe is None:

                continue

            if dataframe.empty:

                continue

            head_columns = self.get_head_columns(
                dataframe
            )

            if not head_columns:

                continue

            dataframe = dataframe[
                head_columns
            ].copy()

            dataframe.reset_index(
                drop=True,
                inplace=True
            )

            if dataframe.empty:

                continue

            valid_files.append(
                dataframe
            )

        if not valid_files:

            return

        rng = np.random.default_rng()

        for participant_id in self.participants:

            selected_count = min(
                self.max_videos,
                len(valid_files)
            )

            selected_indices = rng.choice(
                len(valid_files),
                size=selected_count,
                replace=len(valid_files) < self.max_videos
            )

            for video_index, selected_index in enumerate(
                selected_indices,
                start=1
            ):

                dataframe = valid_files[
                    int(selected_index)
                ].copy()

                if dataframe.empty:

                    continue

                dataframe.reset_index(
                    drop=True,
                    inplace=True
                )

                video_id = f"V{video_index}"

                dataframe.insert(
                    0,
                    "Participant_ID",
                    participant_id
                )

                dataframe.insert(
                    1,
                    "Video_ID",
                    video_id
                )

                output_name = (
                    f"head_{participant_id}_{video_id}.csv"
                )

                output_path = os.path.join(
                    self.head_folder,
                    output_name
                )

                dataframe.to_csv(
                    output_path,
                    index=False,
                    encoding="utf-8-sig"
                )

    def find_questionnaire_files(self):

        files_found = []

        if not os.path.isdir(
            self.questionnaire_fpath
        ):

            return files_found

        for root, directories, files in os.walk(
            self.questionnaire_fpath
        ):

            for filename in files:

                if self.is_temporary_file(
                    filename
                ):

                    continue

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension not in [
                    ".csv",
                    ".xlsx",
                    ".xls",
                    ".txt"
                ]:

                    continue

                files_found.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

        return files_found

    def process_questionnaires(self):

        saved = set()

        for path in self.find_questionnaire_files():

            dataframe = self.read_table(
                path
            )

            if dataframe is None:

                continue

            filename = os.path.basename(
                path
            ).lower()

            if (
                "participant" in filename
                and (
                    "profile" in filename
                    or "information" in filename
                )
            ):

                output_name = (
                    "Participant_Profile.csv"
                )

            elif (
                "pre" in filename
                and (
                    "participant" in filename
                    or "exposure" in filename
                )
            ):

                output_name = (
                    "Participant_Profile_PreExposure.csv"
                )

            elif (
                "post" in filename
                or "rating" in filename
                or "questionnaire" in filename
            ):

                output_name = (
                    "Post_Exposure_Ratings.csv"
                )

            else:

                continue

            if output_name in saved:

                continue

            self.save_dataframe(
                dataframe,
                self.questionnaire_folder,
                output_name
            )

            saved.add(
                output_name
            )

    def create_target(self):

        candidates = []

        for root, directories, files in os.walk(
            self.fpath
        ):

            for filename in files:

                if self.is_temporary_file(
                    filename
                ):

                    continue

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension not in [
                    ".csv",
                    ".xlsx",
                    ".xls",
                    ".txt"
                ]:

                    continue

                path = os.path.join(
                    root,
                    filename
                )

                dataframe = self.read_table(
                    path
                )

                if dataframe is None:

                    continue

                columns = {
                    str(column).strip().lower():
                    column
                    for column in dataframe.columns
                }

                if "quad_cat" not in columns:

                    continue

                if not (
                    "id" in columns
                    or "participant_id" in columns
                ):

                    continue

                if not (
                    "trial_num" in columns
                    or "video_id" in columns
                    or "video" in columns
                ):

                    continue

                candidates.append(
                    dataframe
                )

        if not candidates:

            return

        dataframe = candidates[0]

        columns = {
            str(column).strip().lower():
            column
            for column in dataframe.columns
        }

        if "id" in columns:

            id_column = columns[
                "id"
            ]

        else:

            id_column = columns[
                "participant_id"
            ]

        if "trial_num" in columns:

            trial_column = columns[
                "trial_num"
            ]

        elif "video_id" in columns:

            trial_column = columns[
                "video_id"
            ]

        else:

            trial_column = columns[
                "video"
            ]

        quad_column = columns[
            "quad_cat"
        ]

        rows = []

        for _, row in dataframe.iterrows():

            try:

                source_id = int(
                    float(
                        row[id_column]
                    )
                )

            except Exception:

                continue

            if not (
                101 <= source_id <= 135
            ):

                continue

            participant_id = (
                f"P{source_id + 199}"
            )

            video_id = self.get_video_id(
                row[trial_column]
            )

            if video_id is None:

                continue

            quad_value = row[
                quad_column
            ]

            try:

                if pd.isna(
                    quad_value
                ):

                    continue

            except Exception:

                pass

            rows.append({
                "Participant_ID": participant_id,
                "Video_ID": video_id,
                "Quad_Cat": quad_value
            })

        if not rows:

            return

        target = pd.DataFrame(
            rows
        )

        target.drop_duplicates(
            subset=[
                "Participant_ID",
                "Video_ID"
            ],
            keep="first",
            inplace=True
        )

        target.sort_values(
            [
                "Participant_ID",
                "Video_ID"
            ],
            inplace=True
        )

        target.to_csv(
            os.path.join(
                self.target_folder,
                "Quad_Cat.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

    def count_csv(
        self,
        folder
    ):

        if not os.path.isdir(
            folder
        ):

            return 0

        return len([
            filename
            for filename in os.listdir(folder)
            if filename.lower().endswith(
                ".csv"
            )
        ])

    def run(self):

        self.clean_output()

        self.process_dat_files()

        self.process_head_files()

        self.process_questionnaires()

        self.create_target()

        print(
            f"ECG : {self.ecg_folder} | Count : {self.count_csv(self.ecg_folder)}"
        )

        print(
            f"GSR : {self.gsr_folder} | Count : {self.count_csv(self.gsr_folder)}"
        )

        print(
            f"EYE : {self.eye_folder} | Count : {self.count_csv(self.eye_folder)}"
        )

        print(
            f"HEAD : {self.head_folder} | Count : {self.count_csv(self.head_folder)}"
        )

        print(
            f"QUESTIONNAIRE : {self.questionnaire_folder} | Count : {self.count_csv(self.questionnaire_folder)}"
        )

        print(
            f"TARGET : {self.target_folder} | Count : {self.count_csv(self.target_folder)}"
        )

        print(
            "(spath : processed)"
        )



def main():
    industry_1_converter = format_1(
        fpath=r"..\Dataset\Industry_1",
        spath=r"..\Output\01_input_data\Industry_1"
    )

    industry_2_converter = format_2(
        fpath=r"..\Dataset\Industry_2",
        spath=r"..\Output\01_input_data\Industry_2",
        head_fpath=r"..\Output\01_input_data\Industry_1\04_HEAD_Movement"
    )

    print("\n================ INDUSTRY 1 ================\n")
    industry_1_converter.convert_folder_to_csv()

    print("\n================ INDUSTRY 2 ================\n")
    industry_2_converter.run()


if __name__ == "__main__":
    main()
