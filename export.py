import argparse
from exporter import Exporter

parser = argparse.ArgumentParser(description = 'Transform input onnx model to hw')
parser.add_argument('--onnx-model', required = True, type = str, help = 'QONNX model to transform using FINN Compiler')

def main():
	args = parser.parse_args()

	exporter = Exporter(args.onnx_model)
	steps = [exporter.tidy_up, exporter.post_processing, exporter.streamline]

	for step in steps:
		step()

if __name__ == "__main__":
	main()