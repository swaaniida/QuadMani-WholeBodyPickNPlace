#!/usr/bin/env python3
"""Decode all loco-manipulation videos and create a final-frame contact sheet."""
import json
import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw

from run_locomani_demo import SCENARIOS

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/locomani'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    output = OUT / 'side_by_side' if args.quick else OUT
    scenarios = ('tilt_in_place', 'pick_recover', 'pick_drag') if args.quick else SCENARIOS
    width, height = (640, 180) if args.quick else (1280, 360)
    results = {}
    sheet = Image.new('RGB', (2*width, len(scenarios) * (height+20)), 'white')
    draw = ImageDraw.Draw(sheet)
    for column, variant in enumerate(('profiles_plate', 'plate_only')):
        for row, scenario in enumerate(scenarios):
            name = '{}_{}'.format(variant, scenario)
            path = output / (name + '.mp4')
            reader = imageio.get_reader(str(path))
            frames = reader.count_frames()
            meta = reader.get_meta_data()
            last = reader.get_data(frames - 1)
            reader.close()
            if frames < 60 or tuple(last.shape) != (height, width, 3):
                raise RuntimeError('Invalid video {}: frames={}, shape={}'.format(path, frames, last.shape))
            sheet.paste(Image.fromarray(last), (column * width, row * (height+20) + 20))
            draw.text((column * width + 8, row * (height+20) + 3), name, fill='black')
            for suffix in ('_main', '_wrist'):
                stream = imageio.get_reader(str(output / (name+suffix+'.mp4')))
                assert stream.count_frames() == frames
                assert tuple(stream.get_data(frames-1).shape) == (height,width//2,3)
                stream.close()
            results[name] = {
                'frames': frames,
                'fps': meta['fps'],
                'size': list(meta['size']),
                'duration_s': meta.get('duration'),
                'bytes': path.stat().st_size,
            }
    sheet.save(str(output / 'final_frames.png'))
    (output / 'video_validation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
