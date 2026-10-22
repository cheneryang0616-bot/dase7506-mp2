"""Install the pinned official data after verifying every supplied file hash."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
COMMIT = 'bb801f4f2e7629d270df765ab12c333b2dc616bb'
URL = f'https://codeload.github.com/xudongwu-0/DASE7506-MP2/zip/{COMMIT}'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'code/PACKAGE_MANIFEST.json').read_text())
    expected = {name: value for name, value in manifest['files'].items()
                if name.startswith('code/data/')}
    missing = []
    for name, digest in expected.items():
        path = ROOT / name
        if path.exists():
            if sha(path.read_bytes()) != digest:
                raise ValueError(f'Existing file differs from the course release: {name}')
        else:
            missing.append(name)
    if not missing:
        print(f'Official data already present: {len(expected)} hashes match; no download.')
        return
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        prefix = package.namelist()[0].split('/')[0] + '/'
        verified = {}
        for name in missing:
            content = package.read(prefix + name)
            if sha(content) != expected[name]:
                raise ValueError(f'Downloaded file hash mismatch: {name}')
            verified[name] = content
        for name, content in verified.items():
            path = ROOT / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    print(f'Installed {len(verified)} verified official data files from {COMMIT}.')


if __name__ == '__main__':
    main()
