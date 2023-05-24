import os
import soundfile
import shutil
import pathos
import librosa


def _resample_and_copy_all(filename, source_root, target_root, sr=22050):
    source_file = filename
    target_file = os.path.join(target_root, os.path.relpath(filename, source_root))
    if not os.path.exists(os.path.dirname(target_file)):
        os.makedirs(os.path.dirname(target_file))

    if filename[-4:] == ".wav":
        signal, _ = librosa.load(source_file, sr)
        soundfile.write(target_file, signal, sr)
    else:
        shutil.copy(source_file, target_file)


def preprocess_resampling(path, sr=22050):
    processed_path = os.path.join(os.path.dirname(path), os.path.basename(path) + f'_resample_{sr}')
    all_file = []
    for root, _, files in os.walk(path):
        if files:
            for f in files:
                all_file.append(os.path.join(root, f))

    for f in all_file:
        source_file = f
        target_file = os.path.join(processed_path, os.path.relpath(f, path))
        if not os.path.exists(os.path.dirname(target_file)):
            os.makedirs(os.path.dirname(target_file))

    pool = pathos.multiprocessing.ProcessPool()
    pool.map(_resample_and_copy_all, all_file, [path] * len(all_file), [processed_path] * len(all_file), [sr] * len(all_file))


preprocess_resampling('datasets/LibriTTS/train-clean-100')
