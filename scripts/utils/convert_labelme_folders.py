import os
import pathlib
import argparse
import numpy as np
import PIL.Image as Image


def parse_args():
    parser = argparse.ArgumentParser(description='Re-Organize Exported Labelme dataset.')
    parser.add_argument('--src_root',
                        type=str,
                        required=True,
                        help='root path of labelme exported dataset.')
    parser.add_argument('--tar_root',
                        type=str,
                        required=True,
                        help='root path expected to dump dataset.')

    parser.add_argument('--skip_null', dest='skip_null', action='store_true', default=False)
    parser.add_argument('--clean_raw', dest='clean_raw', action='store_true', default=False)
    parser.add_argument('--no_verbose', dest='no_verbose', action='store_true', default=False)
    
    return parser.parse_args()


def log(*arg, **args):
    if verbose: print(*arg, **args)


def filter_dir(root: pathlib.Path, item) -> bool:
    path = root / item
    # label me exported (from json file) folder structure.
    properties = {'label.png', 'img.png', 'label_viz.png', 'label_names.txt'}
    return os.path.isdir(path) and  set(pathlib.os.listdir(path)) == properties


def is_null_mask(path, thr=3000) -> bool:
    # mask (h, w) @ uint8 for class representation
    mask = np.array(Image.open(path))
    print(mask.shape)
    return (mask > 0).sum() < thr


def main():
    args = parse_args()
    global verbose
    verbose = not args.no_verbose
    # read folders src_root
    src, tar = pathlib.Path(args.src_root), pathlib.Path(args.tar_root)
    
    folders = list(filter(lambda x: filter_dir(src, x), pathlib.os.listdir(src)))
    assert folders, f'No files found in {src}.'
    
    pathlib.os.makedirs(tar / 'imgs', exist_ok=True)
    pathlib.os.makedirs(tar / 'anns', exist_ok=True)

    num_skipped = 0
    for i, folder in enumerate(folders):
        if args.skip_null and is_null_mask(f'{src / folder}/label.png'):
            log(f'Skip {src / folder}/label.png as its mask is null.')
            num_skipped += 1
            continue
        
        os.system(f'cp -r {src / folder}/img.png {tar / 'imgs' / folder}.png')
        os.system(f'cp -r {src / folder}/label.png {tar / 'anns' / folder}.png')
        
        log(f'[{i - num_skipped}] {src / folder}/img.png ==> {tar / 'imgs' / folder}.png')
        log(f'[{i - num_skipped}] {src / folder}/label.png ==> {tar / 'anns' / folder}.png')
    
    log(f'Total {len(folders)} files found. '
            f'{len(folders) - num_skipped} files extracted. {num_skipped} files skipped.')
    
    if args.clean_raw:
        for folder in folders:
            os.system(f'rm -r {src / folder}')
        log(f'Clean {len(folders)} raw files(folders).')
        

if __name__ == '__main__':
    main()