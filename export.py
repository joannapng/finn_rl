import argparse
from exporter import Exporter

parser = argparse.ArgumentParser(description = 'Transform input onnx model to hw')
parser.add_argument('--onnx-model', required = True, type = str, help = 'QONNX model to transform using FINN Compiler')

def main():
	args = parser.parse_args()

	exporter = Exporter(args.onnx_model)
	
	steps = [exporter.tidy_up, exporter.pre_processing, exporter.post_processing, exporter.streamline, exporter.hls_conversion, exporter.create_dataflow_partition, exporter.set_folding, exporter.insert_fifos, exporter.generate_hw]

	for step in steps:
		step()

if __name__ == "__main__":
	main()