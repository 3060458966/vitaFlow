import os
import glob
import librosa
import numpy as np

from tqdm import tqdm
from vitaflow.internal import DatasetInterface, HParams


class TEDLiumDataset(DatasetInterface):
    """
    Download the dataset from https://projets-lium.univ-lemans.fr/wp-content/uploads/corpus/TED-LIUM/TEDLIUM_release1.tar.gz
    And unzip as shown below

    .. code-block:: text

        ~/vitaflow/                     #this can be anywhere on your system
            TEDLiumDataset/
                raw_data/
                    train/
                        sph/
                            speaker_1.sph
                            speaker_2.sph ...
                    dev/
                        sph/
                            speaker_1.sph ...
                    test/
                        sph/
                            speaker_1.sph ...

                    processed_data/ #created as part of this class output
                        train/
                           speaker_1/
                                file_1.wav
                                file_2.wav ...
                            speaker_2/
                                file_1.wav
                                file_2.wav ...
                        dev/
                            speaker_1/
                                file_1.wav ...
                            speaker_2/
                                file_1.wav ...
                        test/
                            speaker_1/
                                file_1.wav ...
    """
    def __init__(self, hparams=None):
        DatasetInterface.__init__(self, hparams=hparams)
        self._hparams = HParams(hparams, self.default_hparams())

        self.extract_speech_clips()

    @staticmethod
    def default_hparams():
        """
        .. role:: python(code)
           :language: python

        .. code-block:: python

            {
                "experiment_root_directory" : os.path.expanduser("~") + "/vitaFlow/",
                "experiment_name" : "experiment_name",
                "preprocessed_data_path" : "preprocessed_data",
                "train_data_path" : "train",
                "validation_data_path" : "val",
                "test_data_path" : "test"
            }

        Here:

        "experiment_root_directory" : str
            Root directory where the data is downloaded or copied, also
            acts as the folder for any subsequent experimentation

        "experiment_name" : str
            Name for the current experiment

        "preprocessed_data_path" : str
            Folder path under `experiment_root_directory` where the preprocessed data
            should be stored

        "train_data_path" : str
            Folder path under `experiment_root_directory` where the train data is stored

        "validation_data_path" : str
            Folder path under `experiment_root_directory` where the validation data is stored

        "test_data_path" : str
            Folder path under `experiment_root_directory` where the test data is stored


        :return: A dictionary of hyperparameters with default values
        """
        hparams = DatasetInterface.default_hparams()

        hparams.update({
            "experiment_name": "TEDLiumDataset",
            "num_clips" : 128,
            "sampling_rate" : 16000,
            "duration": 5
        })

        return hparams

    def audio_clip(self,
                   data_dir,
                   num_clips,
                   duration,
                   output_dir,
                   sampling_rate):
        """
        Creates a folder for each *.sph/*.wav file under given data directory and
        clips the original file by given duration along with starting and end time
        :param data_dir:input data folder
        :param num_clips: The number of clips for each speaker
        :param start_time: starting time of the audio from which the clip is sampled
        :param end_time: ending time of the audio from which the clip is sampled
        :param duration: The duration of each clip
        :param output_dir: the output directory
        :param sampling_rate: Sampling rate
        :return:
        """
        speakers = glob.glob(os.path.join(data_dir, "*.sph"))
        speakers.extend(glob.glob(os.path.join(data_dir, "*.wav")))

        for i in tqdm(range(len(speakers)), desc="speakers"):
            file_path = speakers[i]
            # each speaker can have multiple talks, take first one only
            speaker = file_path.split("/")[-1].split(".")[0]
            speaker_dir = os.path.join(output_dir, speaker)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)

                y, _ = librosa.load(file_path, sr=sampling_rate)
                end_time = librosa.get_duration(y=y, sr=sampling_rate)
                for j in range(num_clips):
                    wav_file = os.path.join(speaker_dir, str(j)) + ".wav"
                    k = int(np.random.randint(0, end_time, size=1))
                    librosa.output.write_wav(wav_file,
                                             y[k*sampling_rate : (k+duration)*sampling_rate],
                                             sampling_rate)
            else:
                # print_info("While processing {}".format(file_path))
                # print_warn("Found existing data @ {}".format(speaker_dir))
                # print_info("Skipping!!!")
                pass

    def extract_speech_clips(self):
        # TODO - Create a dict `inputs` and pass the input as **inputs
        self.audio_clip(data_dir=os.path.join(self.TRAIN_IN_PATH, "sph"),
                        num_clips=self._hparams.num_clips,
                        duration=self._hparams.duration,
                        output_dir=self.TRAIN_OUT_PATH,
                        sampling_rate=self._hparams.sampling_rate)

        self.audio_clip(data_dir=os.path.join(self.VAL_IN_PATH, "sph"),
                        num_clips=self._hparams.num_clips,
                        duration=self._hparams.duration,
                        output_dir=self.VAL_OUT_PATH,
                        sampling_rate=self._hparams.sampling_rate)

        self.audio_clip(data_dir=os.path.join(self.TEST_IN_PATH, "sph"),
                        num_clips=self._hparams.num_clips,
                        duration=self._hparams.duration,
                        output_dir=self.TEST_OUT_PATH,
                        sampling_rate=self._hparams.sampling_rate)
