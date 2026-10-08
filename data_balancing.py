
import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TimeGAN:

    def __init__(self, sequence_length=50, hidden_dim=16, batch_size=64, learning_rate=0.001, embedder_epochs=20, supervisor_epochs=20, joint_epochs=30, random_state=42):
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedder_epochs = embedder_epochs
        self.supervisor_epochs = supervisor_epochs
        self.joint_epochs = joint_epochs
        self.random_state = random_state
        self.feature_dim = None
        self.embedder = None
        self.recovery = None
        self.generator = None
        self.supervisor = None
        self.discriminator = None
        self.embedder_optimizer = None
        self.supervisor_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        np.random.seed(random_state)
        random.seed(random_state)
        tf.random.set_seed(random_state)

    def build_networks(self, feature_dim):
        self.feature_dim = feature_dim
        self.embedder = self.create_rnn_model(feature_dim, self.hidden_dim)
        self.recovery = self.create_rnn_model(self.hidden_dim, feature_dim)
        self.generator = self.create_rnn_model(feature_dim, self.hidden_dim)
        self.supervisor = self.create_rnn_model(self.hidden_dim, self.hidden_dim)
        self.discriminator = self.create_discriminator(self.hidden_dim)
        self.embedder_optimizer = Adam(learning_rate=self.learning_rate)
        self.supervisor_optimizer = Adam(learning_rate=self.learning_rate)
        self.generator_optimizer = Adam(learning_rate=self.learning_rate)
        self.discriminator_optimizer = Adam(learning_rate=self.learning_rate)

    def create_rnn_model(self, input_dim, output_dim):
        inputs = layers.Input(shape=(self.sequence_length, input_dim))
        x = layers.GRU(self.hidden_dim, return_sequences=True)(inputs)
        x = layers.GRU(self.hidden_dim, return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(output_dim, activation="sigmoid"))(x)
        return Model(inputs, outputs)

    def create_discriminator(self, input_dim):
        inputs = layers.Input(shape=(self.sequence_length, input_dim))
        x = layers.GRU(self.hidden_dim, return_sequences=True)(inputs)
        x = layers.GRU(self.hidden_dim, return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(1))(x)
        return Model(inputs, outputs)

    def reconstruction_loss(self, real_data, reconstructed_data):
        return tf.reduce_mean(tf.abs(real_data - reconstructed_data))

    def supervised_loss(self, hidden_data, supervised_data):
        return tf.reduce_mean(tf.square(hidden_data[:, 1:, :] - supervised_data[:, :-1, :]))

    def train_embedder(self, data):
        print()
        print("TimeGAN Embedder Training")
        for epoch in range(self.embedder_epochs):
            indexes = np.random.randint(0, len(data), self.batch_size)
            batch = tf.convert_to_tensor(data[indexes], dtype=tf.float32)
            with tf.GradientTape() as tape:
                hidden = self.embedder(batch, training=True)
                reconstructed = self.recovery(hidden, training=True)
                loss = self.reconstruction_loss(batch, reconstructed)
            variables = self.embedder.trainable_variables + self.recovery.trainable_variables
            gradients = tape.gradient(loss, variables)
            gradients = [(g, v) for g, v in zip(gradients, variables) if g is not None]
            self.embedder_optimizer.apply_gradients(gradients)
            if (epoch + 1) % max(1, self.embedder_epochs // 5) == 0:
                print(f"Embedder Epoch {epoch + 1}/{self.embedder_epochs} | Loss: {float(loss):.6f}")

    def train_supervisor(self, data):
        print()
        print("TimeGAN Supervisor Training")
        for epoch in range(self.supervisor_epochs):
            indexes = np.random.randint(0, len(data), self.batch_size)
            batch = tf.convert_to_tensor(data[indexes], dtype=tf.float32)
            with tf.GradientTape() as tape:
                hidden = self.embedder(batch, training=False)
                supervised = self.supervisor(hidden, training=True)
                loss = self.supervised_loss(hidden, supervised)
            variables = self.supervisor.trainable_variables
            gradients = tape.gradient(loss, variables)
            gradients = [(g, v) for g, v in zip(gradients, variables) if g is not None]
            self.supervisor_optimizer.apply_gradients(gradients)
            if (epoch + 1) % max(1, self.supervisor_epochs // 5) == 0:
                print(f"Supervisor Epoch {epoch + 1}/{self.supervisor_epochs} | Loss: {float(loss):.6f}")

    def train_joint(self, data):
        print()
        print("TimeGAN Generator and Discriminator Training")
        for epoch in range(self.joint_epochs):
            indexes = np.random.randint(0, len(data), self.batch_size)
            real_batch = tf.convert_to_tensor(data[indexes], dtype=tf.float32)
            z = tf.random.uniform((self.batch_size, self.sequence_length, self.feature_dim), 0.0, 1.0)

            with tf.GradientTape() as generator_tape:
                fake_hidden = self.generator(z, training=True)
                supervised_hidden = self.supervisor(fake_hidden, training=True)
                fake_output = self.discriminator(supervised_hidden, training=True)
                generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_output), logits=fake_output))
                real_hidden = self.embedder(real_batch, training=False)
                supervisor_prediction = self.supervisor(real_hidden, training=True)
                supervisor_loss = self.supervised_loss(real_hidden, supervisor_prediction)
                total_generator_loss = generator_loss + supervisor_loss

            generator_variables = self.generator.trainable_variables + self.supervisor.trainable_variables
            generator_gradients = generator_tape.gradient(total_generator_loss, generator_variables)
            generator_gradients = [(g, v) for g, v in zip(generator_gradients, generator_variables) if g is not None]
            self.generator_optimizer.apply_gradients(generator_gradients)

            with tf.GradientTape() as discriminator_tape:
                real_hidden = self.embedder(real_batch, training=False)
                fake_hidden = self.generator(z, training=False)
                supervised_hidden = self.supervisor(fake_hidden, training=False)
                real_output = self.discriminator(real_hidden, training=True)
                fake_output = self.discriminator(supervised_hidden, training=True)
                real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_output), logits=real_output))
                fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_output), logits=fake_output))
                discriminator_loss = real_loss + fake_loss

            discriminator_variables = self.discriminator.trainable_variables
            discriminator_gradients = discriminator_tape.gradient(discriminator_loss, discriminator_variables)
            discriminator_gradients = [(g, v) for g, v in zip(discriminator_gradients, discriminator_variables) if g is not None]
            self.discriminator_optimizer.apply_gradients(discriminator_gradients)

            if (epoch + 1) % max(1, self.joint_epochs // 5) == 0:
                print(f"Joint Epoch {epoch + 1}/{self.joint_epochs} | Generator Loss: {float(total_generator_loss):.6f} | Discriminator Loss: {float(discriminator_loss):.6f}")

    def fit(self, data):
        if data is None or len(data) == 0:
            raise ValueError("TimeGAN training data is empty")
        self.build_networks(data.shape[-1])
        self.train_embedder(data)
        self.train_supervisor(data)
        self.train_joint(data)
        print()
        print("TimeGAN Training Completed")

    def generate(self, number_of_sequences):
        generated_sequences = []
        remaining = number_of_sequences
        while remaining > 0:
            current_batch = min(self.batch_size, remaining)
            z = tf.random.uniform((current_batch, self.sequence_length, self.feature_dim), 0.0, 1.0)
            fake_hidden = self.generator(z, training=False)
            supervised_hidden = self.supervisor(fake_hidden, training=False)
            synthetic = self.recovery(supervised_hidden, training=False)
            synthetic = np.clip(synthetic.numpy(), 0.0, 1.0)
            generated_sequences.append(synthetic)
            remaining -= current_batch
        return np.concatenate(generated_sequences, axis=0)


class ParticipantDataBalancer:

    def __init__(self, input_path, output_path, industry_name, industry_id, original_participant_count=32, augmentation_participant_count=40, files_per_participant=41, random_state=42, sequence_length=50, hidden_dim=16, batch_size=64, embedder_epochs=20, supervisor_epochs=20, joint_epochs=30):
        self.input_path = input_path
        self.output_path = output_path
        self.industry_name = industry_name
        self.industry_id = industry_id
        self.original_participant_count = original_participant_count
        self.augmentation_participant_count = augmentation_participant_count
        self.files_per_participant = files_per_participant
        self.target_participant_count = original_participant_count + augmentation_participant_count
        self.random_state = random_state
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedder_epochs = embedder_epochs
        self.supervisor_epochs = supervisor_epochs
        self.joint_epochs = joint_epochs
        self.modalities = {"01_ECG": "ecg", "02_GSR": "gsr", "03_EYE_Movement": "EM", "04_HEAD_Movement": "HM"}
        self.questionnaire_folder = "05_Questionnaires"
        self.participant_information_file = "participant_information.csv"
        self.source_participant_mapping = {}
        self.modality_feature_columns = {}
        self.modality_scalers = {}
        random.seed(random_state)
        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def run(self):
        input_industry_path = os.path.join(self.input_path, self.industry_name)
        output_industry_path = os.path.join(self.output_path, self.industry_name)
        os.makedirs(output_industry_path, exist_ok=True)

        for folder_name in self.modalities:
            os.makedirs(os.path.join(output_industry_path, folder_name), exist_ok=True)

        os.makedirs(os.path.join(output_industry_path, self.questionnaire_folder), exist_ok=True)

        print()
        print("============================================================")
        print(f"Industry              : {self.industry_name}")
        print(f"Input                 : {input_industry_path}")
        print(f"Output                : {output_industry_path}")
        print(f"Original Participants : {self.original_participant_count}")
        print(f"New Augmented         : {self.augmentation_participant_count}")
        print(f"Final Participants    : {self.target_participant_count}")
        print(f"Files Per Participant : {self.files_per_participant}")
        print("============================================================")

        self.create_source_participant_mapping()
        questionnaire_data = self.load_questionnaires(input_industry_path)

        for folder_name, prefix in self.modalities.items():
            self.process_modality(input_industry_path, output_industry_path, folder_name, prefix)

        self.create_augmented_questionnaires(output_industry_path, questionnaire_data)
        self.create_augmented_participant_information(input_industry_path, output_industry_path)

        print()
        print("============================================================")
        print(f"{self.industry_name} Augmentation Completed")
        print(f"P1-P{self.target_participant_count}")
        print("============================================================")

    def create_source_participant_mapping(self):
        source_participants = list(range(1, self.original_participant_count + 1))
        self.source_participant_mapping = {}
        for participant_number in range(self.original_participant_count + 1, self.target_participant_count + 1):
            self.source_participant_mapping[participant_number] = random.choice(source_participants)

        print()
        print("Source Participant Mapping")
        for participant_number, source_participant in self.source_participant_mapping.items():
            print(f"P{participant_number} -> P{source_participant}")

    def process_modality(self, input_industry_path, output_industry_path, folder_name, prefix):
        input_folder = os.path.join(input_industry_path, folder_name)
        output_folder = os.path.join(output_industry_path, folder_name)

        if not os.path.isdir(input_folder):
            print(f"{folder_name}: Input folder not found")
            return

        participant_files = self.group_files_by_participant(input_folder)
        original_participants = {}

        for participant_number in range(1, self.original_participant_count + 1):
            files = sorted(participant_files.get(participant_number, []), key=self.get_version_number)
            if not files:
                continue

            participant_data = []
            for file_path in files:
                df = self.read_file(file_path)
                if df is not None and not df.empty:
                    participant_data.append(df)

            if participant_data:
                original_participants[participant_number] = participant_data

        if not original_participants:
            print(f"{folder_name}: No valid participant data")
            return

        print()
        print("============================================================")
        print(f"{self.industry_name} - {folder_name}")
        print("============================================================")

        for participant_number in sorted(original_participants):
            participant_data = original_participants[participant_number]
            for version, df in enumerate(participant_data, 1):
                output_name = self.create_output_filename(prefix, participant_number, version)
                df.to_csv(os.path.join(output_folder, output_name), index=False)
            print(f"{prefix} P{participant_number}: {len(participant_data)} original files")

        training_data = self.prepare_timegan_training_data(original_participants, folder_name)

        if training_data is None:
            print(f"{folder_name}: TimeGAN training skipped")
            return

        print()
        print(f"TimeGAN Data: {folder_name}")
        print(f"Sequences : {training_data.shape[0]}")
        print(f"Length    : {training_data.shape[1]}")
        print(f"Features  : {training_data.shape[2]}")

        timegan = TimeGAN(sequence_length=self.sequence_length, hidden_dim=self.hidden_dim, batch_size=self.batch_size, embedder_epochs=self.embedder_epochs, supervisor_epochs=self.supervisor_epochs, joint_epochs=self.joint_epochs, random_state=self.random_state)
        timegan.fit(training_data)

        generated_file_count = self.augmentation_participant_count * self.files_per_participant

        print()
        print(f"Generating {generated_file_count} synthetic files")

        generated_sequences = timegan.generate(generated_file_count)
        self.generate_augmented_files(generated_sequences, original_participants, output_folder, prefix, folder_name)

    def prepare_timegan_training_data(self, original_participants, folder_name):
        feature_set = set()

        for participant_number in sorted(original_participants):
            for df in original_participants[participant_number]:
                feature_set.update(self.get_numeric_columns(df))

        if not feature_set:
            print(f"{folder_name}: No numeric columns")
            return None

        feature_columns = sorted(feature_set)
        self.modality_feature_columns[folder_name] = feature_columns

        print()
        print(f"Common Features: {folder_name}")
        print(f"Feature Count: {len(feature_columns)}")

        global_values = {column: [] for column in feature_columns}

        for participant_number in sorted(original_participants):
            for df in original_participants[participant_number]:
                for column in feature_columns:
                    if column not in df.columns:
                        continue
                    values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) > 0:
                        global_values[column].extend(values.to_numpy(dtype=np.float32).tolist())

        fill_values = {}
        minimum = []
        maximum = []

        for column in feature_columns:
            values = global_values[column]
            if values:
                values = np.asarray(values, dtype=np.float32)
                fill_values[column] = float(np.median(values))
                minimum.append(float(np.min(values)))
                maximum.append(float(np.max(values)))
            else:
                fill_values[column] = 0.0
                minimum.append(0.0)
                maximum.append(1.0)

        minimum = np.asarray(minimum, dtype=np.float32)
        maximum = np.asarray(maximum, dtype=np.float32)
        denominator = maximum - minimum
        denominator[denominator == 0] = 1.0

        self.modality_scalers[folder_name] = (minimum, maximum, denominator, fill_values)
        sequences = []

        for participant_number in sorted(original_participants):
            for df in original_participants[participant_number]:
                aligned = pd.DataFrame(index=df.index)

                for column in feature_columns:
                    if column in df.columns:
                        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(fill_values[column])
                        aligned[column] = values
                    else:
                        aligned[column] = fill_values[column]

                if len(aligned) < 2:
                    continue

                values = aligned.to_numpy(dtype=np.float32)
                values = self.resample_sequence(values, self.sequence_length)
                values = (values - minimum) / denominator
                values = np.clip(values, 0.0, 1.0)
                values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
                sequences.append(values.astype(np.float32))

        if not sequences:
            return None

        data = np.stack(sequences, axis=0)
        print(f"Final TimeGAN Shape: {data.shape}")
        return data

    def generate_augmented_files(self, generated_sequences, original_participants, output_folder, prefix, folder_name):
        sequence_index = 0
        available_sources = list(original_participants.keys())

        print()
        print("Synthetic Participant Generation")

        for participant_number in range(self.original_participant_count + 1, self.target_participant_count + 1):
            source_participant = self.source_participant_mapping[participant_number]
            if source_participant not in original_participants:
                source_participant = random.choice(available_sources)

            source_data = original_participants[source_participant]
            generated_count = 0

            for version in range(1, self.files_per_participant + 1):
                source_index = (version - 1) % len(source_data)
                source_df = source_data[source_index]
                generated_sequence = generated_sequences[sequence_index]
                augmented_df = self.create_generated_dataframe(generated_sequence, source_df, folder_name)
                output_name = self.create_output_filename(prefix, participant_number, version)
                augmented_df.to_csv(os.path.join(output_folder, output_name), index=False)
                sequence_index += 1
                generated_count += 1

            print(f"{prefix} P{participant_number}: {generated_count} augmented files | Source P{source_participant}")

    def create_generated_dataframe(self, generated_sequence, source_df, folder_name):
        result = source_df.copy()
        feature_columns = self.modality_feature_columns[folder_name]
        minimum, maximum, denominator, fill_values = self.modality_scalers[folder_name]
        generated_sequence = self.resample_sequence(generated_sequence, len(source_df))
        numeric_columns = self.get_numeric_columns(source_df)

        for column in numeric_columns:
            if column not in feature_columns:
                continue

            column_index = feature_columns.index(column)
            generated_normalized = generated_sequence[:, column_index]
            generated_global = generated_normalized * denominator[column_index] + minimum[column_index]

            source_values = pd.to_numeric(source_df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").fillna(fill_values[column]).to_numpy(dtype=np.float32)

            source_min = np.min(source_values)
            source_max = np.max(source_values)
            generated_min = np.min(generated_global)
            generated_max = np.max(generated_global)

            if source_max == source_min or generated_max == generated_min:
                generated_values = np.full(len(source_df), source_min, dtype=np.float32)
            else:
                generated_values = ((generated_global - generated_min) / (generated_max - generated_min)) * (source_max - source_min) + source_min

            result[column] = generated_values

        return result

    def create_output_filename(self, prefix, participant_number, version):
        return f"{self.industry_id}_{prefix}_P{participant_number}_V{version}.csv"

    def group_files_by_participant(self, folder):
        participant_files = {}
        pattern = re.compile(r"_P(\d+)(?:_V(\d+))?", re.IGNORECASE)

        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(file_name)[1].lower()
            if extension not in [".csv", ".txt", ".xlsx", ".xls"]:
                continue

            match = pattern.search(os.path.splitext(file_name)[0])
            if not match:
                continue

            participant_number = int(match.group(1))
            participant_files.setdefault(participant_number, []).append(file_path)

        return participant_files

    def get_version_number(self, file_path):
        match = re.search(r"_V(\d+)", os.path.basename(file_path), re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def get_numeric_columns(self, df):
        return list(df.select_dtypes(include=[np.number]).columns)

    def resample_sequence(self, values, target_length):
        current_length = values.shape[0]

        if current_length == target_length:
            return values.astype(np.float32)

        old_index = np.linspace(0.0, 1.0, current_length)
        new_index = np.linspace(0.0, 1.0, target_length)
        result = np.zeros((target_length, values.shape[1]), dtype=np.float32)

        for column_index in range(values.shape[1]):
            result[:, column_index] = np.interp(new_index, old_index, values[:, column_index])

        return result

    def read_file(self, file_path):
        try:
            extension = os.path.splitext(file_path)[1].lower()

            if extension == ".csv":
                df = pd.read_csv(file_path)
            elif extension == ".txt":
                try:
                    df = pd.read_csv(file_path, sep=None, engine="python")
                except Exception:
                    df = pd.read_csv(file_path, header=None)
            elif extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            else:
                return None

            df.columns = [str(column).strip() for column in df.columns]
            return df

        except Exception as e:
            print(f"Error reading file: {file_path} -> {e}")
            return None

    def load_questionnaires(self, input_industry_path):
        questionnaire_path = os.path.join(input_industry_path, self.questionnaire_folder)
        questionnaire_data = {}

        if not os.path.isdir(questionnaire_path):
            print("Questionnaire folder not found:", questionnaire_path)
            return questionnaire_data

        for file_name in sorted(os.listdir(questionnaire_path)):
            file_path = os.path.join(questionnaire_path, file_name)

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(file_name)[1].lower()
            if extension not in [".csv", ".txt", ".xlsx", ".xls"]:
                continue

            df = self.read_questionnaire_file(file_path)
            if df is not None and not df.empty:
                questionnaire_data[file_name] = df

        return questionnaire_data

    def read_questionnaire_file(self, file_path):
        try:
            extension = os.path.splitext(file_path)[1].lower()

            if extension == ".csv":
                df = pd.read_csv(file_path, header=None, dtype=object)
            elif extension == ".txt":
                df = pd.read_csv(file_path, header=None, dtype=object, sep=None, engine="python")
            elif extension in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, header=None, dtype=object)
            else:
                return None

            df = df.dropna(how="all")

            if df.empty:
                return None

            participant_row = None

            for row_index in range(len(df)):
                values = df.iloc[row_index].astype(str).str.strip().str.lower().tolist()
                if "participant_id" in values or "participant id" in values:
                    participant_row = row_index
                    break

            if participant_row is None:
                return df

            header = df.iloc[participant_row].astype(str).str.strip().tolist()
            data = df.iloc[participant_row + 1:].copy()
            data.columns = header
            return data.reset_index(drop=True)

        except Exception as e:
            print(f"Error reading questionnaire: {file_path} -> {e}")
            return None

    def create_augmented_questionnaires(self, output_industry_path, questionnaire_data):
        output_folder = os.path.join(output_industry_path, self.questionnaire_folder)

        for file_name, df in questionnaire_data.items():
            participant_column = self.find_participant_column(df)

            if participant_column is None:
                output_name = self.create_questionnaire_filename(file_name)
                df.to_csv(os.path.join(output_folder, output_name), index=False)
                continue

            source_rows = {}

            for participant_number in range(1, self.original_participant_count + 1):
                participant_id = f"P{participant_number}"
                matched = self.get_participant_rows(df, participant_column, participant_id)
                if not matched.empty:
                    source_rows[participant_number] = matched.iloc[0].copy()

            if not source_rows:
                continue

            output_rows = []

            for participant_number in range(1, self.target_participant_count + 1):
                if participant_number <= self.original_participant_count:
                    if participant_number not in source_rows:
                        continue
                    new_row = source_rows[participant_number].copy()
                else:
                    source_participant = self.source_participant_mapping[participant_number]
                    if source_participant not in source_rows:
                        source_participant = random.choice(list(source_rows.keys()))
                    new_row = self.augment_questionnaire_values(source_rows[source_participant])

                new_row[participant_column] = f"P{participant_number}"
                output_rows.append(new_row)

            output_df = pd.DataFrame(output_rows)
            output_name = self.create_questionnaire_filename(file_name)
            output_df.to_csv(os.path.join(output_folder, output_name), index=False)
            print(f"Questionnaire: {output_name}")

    def augment_questionnaire_values(self, row):
        augmented = row.copy()

        for column in augmented.index:
            normalized = str(column).strip().lower().replace(" ", "_")

            if normalized in ["participant_id", "participant", "participantid", "id"]:
                continue

            value = augmented[column]

            if pd.isna(value):
                continue

            numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

            if pd.isna(numeric_value):
                continue

            if -3 <= numeric_value <= 3:
                noise = random.choice([-1, 0, 0, 0, 1])
                augmented[column] = int(round(max(-3, min(3, numeric_value + noise))))

        return augmented

    def create_questionnaire_filename(self, file_name):
        base_name = os.path.splitext(file_name)[0]
        base_name = re.sub(r"^ID_\d+_", "", base_name, flags=re.IGNORECASE)
        return f"{self.industry_id}_{base_name}.csv"

    def create_augmented_participant_information(self, input_industry_path, output_industry_path):
        input_file = os.path.join(input_industry_path, self.participant_information_file)

        if not os.path.isfile(input_file):
            print("Participant information not found:", input_file)
            return

        df = self.read_file(input_file)

        if df is None or df.empty:
            return

        participant_column = self.find_participant_column(df)

        if participant_column is None:
            print("Participant ID column not found")
            return

        source_rows = {}

        for participant_number in range(1, self.original_participant_count + 1):
            participant_id = f"P{participant_number}"
            matched = self.get_participant_rows(df, participant_column, participant_id)
            if not matched.empty:
                source_rows[participant_number] = matched.iloc[0].copy()

        if not source_rows:
            return

        output_rows = []

        for participant_number in range(1, self.target_participant_count + 1):
            if participant_number <= self.original_participant_count:
                if participant_number not in source_rows:
                    continue
                new_row = source_rows[participant_number].copy()
            else:
                source_participant = self.source_participant_mapping[participant_number]
                if source_participant not in source_rows:
                    source_participant = random.choice(list(source_rows.keys()))
                new_row = source_rows[source_participant].copy()

            new_row[participant_column] = f"P{participant_number}"
            output_rows.append(new_row)

        output_df = pd.DataFrame(output_rows)

        output_file = os.path.join(output_industry_path, f"{self.industry_id}_participant_information.csv")
        output_df.to_csv(output_file, index=False)

        print(f"Participant Information: P1-P{self.target_participant_count}")

    def find_participant_column(self, df):
        possible_columns = ["Participant_ID", "Participant_id", "ParticipantID", "Participant Id", "Participant", "participant_id", "participantid", "participant", "ID", "Id"]

        normalized_columns = {str(column).strip().lower().replace(" ", "_").replace("-", "_"): column for column in df.columns}

        for column in possible_columns:
            normalized = column.lower().replace(" ", "_").replace("-", "_")
            if normalized in normalized_columns:
                return normalized_columns[normalized]

        return None

    def get_participant_rows(self, df, participant_column, participant_id):
        values = df[participant_column].astype(str).str.strip().str.upper()
        return df[values == participant_id.upper()]


def run_industry(config):
    balancer = ParticipantDataBalancer(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        industry_name=config["industry_name"],
        industry_id=config["industry_id"],
        original_participant_count=config["original_participant_count"],
        augmentation_participant_count=config["augmentation_participant_count"],
        files_per_participant=config["files_per_participant"],
        random_state=config["random_state"],
        sequence_length=TIMEGAN_SEQUENCE_LENGTH,
        hidden_dim=TIMEGAN_HIDDEN_DIM,
        batch_size=TIMEGAN_BATCH_SIZE,
        embedder_epochs=TIMEGAN_EMBEDDER_EPOCHS,
        supervisor_epochs=TIMEGAN_SUPERVISOR_EPOCHS,
        joint_epochs=TIMEGAN_JOINT_EPOCHS
    )
    balancer.run()


if __name__ == "__main__":

    INPUT_PATH = r"..\Output\01_input_data"
    OUTPUT_PATH = r"..\Output\02_augumentation"

    ORIGINAL_PARTICIPANT_COUNT = 32
    AUGMENTATION_PARTICIPANT_COUNT = 33

    INDUSTRY_1_FILES_PER_PARTICIPANT = 32
    INDUSTRY_2_FILES_PER_PARTICIPANT = 12

    RANDOM_STATE = 42

    TIMEGAN_SEQUENCE_LENGTH = 50
    TIMEGAN_HIDDEN_DIM = 16
    TIMEGAN_BATCH_SIZE = 64
    TIMEGAN_EMBEDDER_EPOCHS = 20
    TIMEGAN_SUPERVISOR_EPOCHS = 20
    TIMEGAN_JOINT_EPOCHS = 30

    INDUSTRY_CONFIGS = [
        {"industry_name": "Industry_1", "industry_id": "ID_1", "original_participant_count": 32, "augmentation_participant_count": AUGMENTATION_PARTICIPANT_COUNT, "files_per_participant": INDUSTRY_1_FILES_PER_PARTICIPANT, "random_state": RANDOM_STATE},
        {"industry_name": "Industry_2", "industry_id": "ID_2", "original_participant_count": 32, "augmentation_participant_count": AUGMENTATION_PARTICIPANT_COUNT, "files_per_participant": INDUSTRY_2_FILES_PER_PARTICIPANT, "random_state": RANDOM_STATE + 1}
    ]

    print()
    print("============================================================")
    print("MULTI-INDUSTRY TIMEGAN AUGMENTATION")
    print("============================================================")
    print(f"Original Participants : {ORIGINAL_PARTICIPANT_COUNT}")
    print(f"New Participants      : {AUGMENTATION_PARTICIPANT_COUNT}")
    print(f"Final Participants    : {ORIGINAL_PARTICIPANT_COUNT + AUGMENTATION_PARTICIPANT_COUNT}")
    print(f"Industry_1 Files      : {INDUSTRY_1_FILES_PER_PARTICIPANT}")
    print(f"Industry_2 Files      : {INDUSTRY_2_FILES_PER_PARTICIPANT}")
    print(f"Output                : {OUTPUT_PATH}")
    print("============================================================")

    for config in INDUSTRY_CONFIGS:
        run_industry(config)

    print()
    print("============================================================")
    print("ALL AUGMENTATION COMPLETED")
    print("============================================================")
