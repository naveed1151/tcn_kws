import os
import requests
import tarfile

def download_and_extract():
    url = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    target_dir = './data'
    filename = os.path.join(target_dir, 'speech_commands_v0.02.tar.gz')

    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(filename):
        print("Downloading GSCv2...")
        r = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    print("Extracting...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(path=target_dir)

if __name__ == "__main__":
    download_and_extract()