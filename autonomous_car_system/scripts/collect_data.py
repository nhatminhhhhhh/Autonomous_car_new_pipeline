import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from data.teleoperation import TeleoperationCollector
from configs.config import CONFIG


def main():
    parser = argparse.ArgumentParser(description='Collect driving data')
    parser.add_argument('--save-dir', type=str, default=CONFIG['data_dir'],
                        help='Directory to save frames and labels')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Saving driving data to: {args.save_dir}")

    collector = TeleoperationCollector(save_dir=args.save_dir, cam_index=args.camera)
    collector.run()


if __name__ == '__main__':
    main()
