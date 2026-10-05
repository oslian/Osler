"""Verify release assets, then optionally restore safe paths without overwriting."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    root = Path.cwd().resolve()
    assets = json.loads((root / 'RESEARCH_ASSETS.json').read_text())['assets']
    weights = json.loads((root / 'RESEARCH_WEIGHTS_MANIFEST.json').read_text())
    records = {x['path']: x for x in weights['files'] + weights['reader_files']}
    for asset in assets:
        archive = args.asset_dir / asset['name']
        if archive.stat().st_size != asset['bytes'] or digest(archive) != asset['sha256']:
            raise ValueError('Archive checksum mismatch: ' + archive.name)
        seen = set()
        with tarfile.open(archive, 'r:gz') as tar:
            for member in tar:
                name = PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or not member.isfile():
                    raise ValueError('Unsafe member: ' + member.name)
                if member.name not in asset['files'] or member.name in seen:
                    raise ValueError('Unexpected/duplicate member: ' + member.name)
                if member.size != records[member.name]['bytes']:
                    raise ValueError('Member size mismatch: ' + member.name)
                seen.add(member.name)
                destination = root.joinpath(*name.parts)
                if not destination.resolve().is_relative_to(root) or destination.is_symlink():
                    raise ValueError('Unsafe destination: ' + member.name)
                if args.extract and destination.exists():
                    if digest(destination) != records[member.name]['sha256']:
                        raise ValueError('Refusing overwrite: ' + member.name)
                    continue
                stream = tar.extractfile(member)
                h = hashlib.sha256()
                temporary = None
                output = None
                try:
                    if args.extract:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        pending = destination.with_name(destination.name + '.download-tmp')
                        output = pending.open('xb')
                        temporary = pending
                    while block := stream.read(1024 * 1024):
                        h.update(block)
                        if output is not None:
                            output.write(block)
                    if output is not None:
                        output.close()
                        output = None
                    if h.hexdigest() != records[member.name]['sha256']:
                        raise ValueError('Member checksum mismatch: ' + member.name)
                    if temporary is not None:
                        temporary.replace(destination)
                finally:
                    if output is not None:
                        output.close()
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
        if seen != set(asset['files']):
            raise ValueError('Missing members: ' + archive.name)
        print('Verified' + (' and restored' if args.extract else ''), archive.name)


if __name__ == '__main__':
    main()
