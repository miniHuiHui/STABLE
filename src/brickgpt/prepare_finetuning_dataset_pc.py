import os
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from datasets import load_dataset
from transformers import HfArgumentParser
from brickgpt.data.brick_structure import BrickStructure

def create_pc_instruction(point_cloud: str) -> str:
    instruction = ('Create a LEGO model of the input 3D point cloud. Format your response as a list of bricks: '
                   '<brick dimensions> <brick position>, where the brick position is (x,y,z).\n'
                   'Allowed brick dimensions are 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n'
                   'All bricks are 1 unit tall.\n\n'
                   '### Input Point Cloud:\n'
                   f'{point_cloud}')
    return instruction


@dataclass
class PreparePCDatasetArguments:
    input_path: str = field(
        default='AvaLovelace/StableText2Brick',
        metadata={'help': 'Path to the directory containing the brick structure dataset to be processed.'},
    )
    output_path: str = field(
        default='datasets_pc',
        metadata={'help': 'Path to save process dataset.'},
    )


def brick_to_point_cloud(bricks_txt: str) -> str:
    structure = BrickStructure.from_txt(bricks_txt)
    points = []
    for x in range(structure.world_dim):
        for y in range(structure.world_dim):
            for z in range(structure.world_dim):
                if structure.voxel_occupancy[x, y, z] > 0:
                    points.append(f'({x},{y},{z})')
    return ', '.join(points)

def main():
    parser = HfArgumentParser(PreparePCDatasetArguments)
    (cfg,) = parser.parse_args_into_dataclasses()

    input_dataset = load_dataset(cfg.input_path)

    def convert_sample(batch: MutableMapping) -> dict:
        messages_list = []
        for bricks in batch['bricks']:
            try:
                pc_str = brick_to_point_cloud(bricks)
            except Exception:
                pc_str = ""
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': create_pc_instruction(pc_str)},
                {'role': 'assistant', 'content': bricks},
            ]
            messages_list.append(messages)
        return {'messages': messages_list}

    os.makedirs(cfg.output_path, exist_ok=True)
    for split_name, split in input_dataset.items():
        output_split = split.map(
            convert_sample,
            batched=True,
            remove_columns=split.column_names,
            desc=f'Converting dataset split "{split_name}"',
        )
        output_split.to_json(Path(cfg.output_path) / f'{split_name}.jsonl')

    print(f'Converted dataset saved to {os.path.abspath(cfg.output_path)}')


if __name__ == '__main__':
    main()
