from ultralytics import YOLO
import argparse
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model with custom parameters.")
    parser.add_argument('--model', '-m', type=str, default="yolo11n.pt", help="Path to the YOLO model file.")
    parser.add_argument('--data', type=str, default="data.yaml", help="Path to the dataset YAML file.")
    parser.add_argument('--batch', type=float, default=0.9, help="Batch size percentage for training.")
    parser.add_argument('--cache', type=bool, default=True, help="Whether to cache images for faster training.")
    parser.add_argument('--time', '-t', type=float, required=True, help="Training time limit (in hours).")
    time_str = str(datetime.now().strftime("%Y%m%d_hour%H_min%M"))
    parser.add_argument('--name', type=str, default=time_str, help="Name to save training results with timestamp.")
    parser.add_argument('--project', '-P', type=str, required=True, help="Project to save training results with timestamp.")
    parser.add_argument('--resume', '-R', type=str, default=False, help="Whether continue to train from un_finished model")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Training YOLO model with the following parameters:")
    print(f"  Model:   {args.model}")
    print(f"  Data:    {args.data}")
    print(f"  Batch:   {args.batch}")
    print(f"  Cache:   {args.cache}")
    print(f"  Time:    {args.time}")
    print(f"  Name:    {args.name}")
    print(f"  Project: {args.project}")
    print(f"  Resume:  {args.resume}")

    model = YOLO(model=args.model, verbose=False)
    model.train(
        data=args.data,
        batch=args.batch,
        cache=args.cache,
        time=args.time,
        name=args.name,
        resume=args.resume,
        profile=True,
        exist_ok=True,
        project=f"train_results/{args.project}"
    )

if __name__ == '__main__':
    main()
