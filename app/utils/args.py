import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="openai/whisper-large-v3")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--root-path", type=str, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()
