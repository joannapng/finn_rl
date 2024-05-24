import numpy as np
import qonnx.custom_op.registry as registry
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys

def set_defaults(model):
	for node in model.graph.node:
		inst = registry.getCustomOp(node)
		attrs = inst.get_nodeattr_types()

		if "PE" in attrs:
			inst.set_nodeattr("PE", 1)
		
		if "SIMD" in attrs:
			inst.set_nodeattr("SIMD", 1)

def estimate_resources(model):
	res_dict = {}
	for node in model.graph.node:
		if is_hls_node(node) or is_rtl_node(node):
			inst =  registry.getCustomOp(node)
			res_dict[node.name] = inst.node_res_estimation()
	
	return res_dict

def reduceBRAMUsage(model, resources_per_layer):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['BRAM_18K'], reverse = True)

	for layer in sorted_resources_per_layer:
		name, _ = layer

		node = model.get_node_from_name(name)
		node_inst = registry.getCustomOp(node)
		op_type = node.op_type
		
		if op_type in ["MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl"]:
			mem_mode = node_inst.get_nodeattr("mem_mode")
			if mem_mode != "internal_embedded" and node_inst.calc_wmem() <= 128:
				node_inst.set_nodeattr("mem_mode", "internal_embedded")
				break
			elif mem_mode == "internal_decoupled":
				node_inst.set_nodeattr("ram_style", "ultra")
				if node_inst.uram_efficiency_estimation() < 0.1:
					node_inst.set_nodeattr("ram_style", "distributed")
					break
		elif op_type == "Channelwise_op_hls":
			ram_style = node_inst.get_nodeattr("ram_style")
			tmem = node_inst.calc_tmem()
			if ram_style == "block" and tmem > 1:
				node_inst.set_nodeattr("ram_style", "distributed")
				break
		elif op_type.startswith("ConvolutionInputGenerator"):
			ram_style = node_inst.get_nodeattr("ram_style")
			if ram_style != "distributed":
				node_inst.set_nodeattr("ram_style", "distributed")
				break
		
	return model

def reduceDSPUsage(model, resources_per_layer):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['DSP'], reverse = True)
	for layer in sorted_resources_per_layer:
		name, _ = layer
		node = model.get_node_from_name(name)
		node_inst = registry.getCustomOp(node)
		op_type = node.op_type
		
		if op_type in ["MVAU_hls", "VVAU_hls"]:
			res_type = node_inst.get_nodeattr("resType")

			if res_type != "lut":
				node_inst.set_nodeattr("resType", "lut")
				break
		
	return model

def reduceLUTUsage(model, resources_per_layer):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['LUT'], reverse = True)
	for layer in sorted_resources_per_layer:
		name, _ = layer
		node = model.get_node_from_name(name)
		node_inst = registry.getCustomOp(node)
		op_type = node.op_type
		
		if op_type in ["MVAU_hls", "VVAU_hls"]:
			res_type = node_inst.get_nodeattr("resType")

			if res_type != "dsp":
				node_inst.set_nodeattr("resType", "dsp")
				break
		elif op_type == "ChannelwiseOp_hls":
			ram_style = node_inst.get_nodeattr("ram_style")
			tmem = node_inst.calc_tmem()

			if ram_style == "distributed" and tmem > 1:
				node_inst.set_nodeattr("ram_style", "block")
				break
		elif op_type.startswith("ConvolutionInputGenerator"):
			ram_style = node_inst.get_nodeattr("ram_style")
			
			if ram_style == "distributed":
				node_inst.set_nodeattr("ram_style", "ultra")
				if node_inst.uram_efficiency_estimation() < 0.1:
					node_inst.set_nodeattr("ram_style", "block")
				break
		elif op_type == "Thresholding_hls":
			ram_style = node_inst.get_nodeattr("ram_style")
			tmem = node_inst.calc_tmem()
			if ram_style == "distributed" and tmem > 1:
				node_inst.set_nodeattr("ram_style", "block")
				break
		
	return model

def reduceURAMUsage(model, resources_per_layer):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['URAM'], reverse = True)
	for layer in sorted_resources_per_layer:
		name, _ = layer

		node = model.get_node_from_name(name)
		node_inst = registry.getCustomOp(node)
		op_type = node.op_type
		
		if op_type in ["MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl"]:
			mem_mode = node_inst.get_nodeattr("mem_mode")
			if mem_mode == "internal_decoupled" and node_inst.get_nodeattr("ram_style") == "ultra":
				if node_inst.bram_efficiency_estimation() >= 0.5:
					node_inst.set_nodeattr("ram_style", "BRAM")
				else:
					node_inst.set_nodeattr("ram_style", "distributed")
				break
		elif op_type == "Channelwise_op_hls":
			ram_style = node_inst.get_nodeattr("ram_style")
			tmem = node_inst.calc_tmem()
			if ram_style == "block" and tmem > 1:
				node_inst.set_nodeattr("ram_style", "distributed")
				break
		elif op_type.startswith("ConvolutionInputGenerator"):
			ram_style = node_inst.get_nodeattr("ram_style")
			if ram_style != "distributed":
				node_inst.set_nodeattr("ram_style", "distributed")
				break
		
	return model	

def isFeasible(model, available_resources, max_iters = 100):
	resources_per_layer = estimate_resources(model)
	resources_total = aggregate_dict_keys(resources_per_layer)
	
	iters = 1
	while iters < max_iters and np.any(np.array(list(resources_total.values())) > np.array(list(available_resources.values()))):		
		iters += 1
		if resources_total['BRAM_18K'] > available_resources['BRAM_18K']:
			model = reduceBRAMUsage(model, resources_per_layer)
		
		if resources_total['LUT'] > available_resources['LUT']:
			model = reduceLUTUsage(model, resources_per_layer)
		
		if resources_total['URAM'] > available_resources['URAM']:
			model = reduceURAMUsage(model, resources_per_layer)

		if resources_total['DSP'] > available_resources['DSP']:
			model = reduceDSPUsage(model, resources_per_layer)

		resources_per_layer = estimate_resources(model)
		resources_total = aggregate_dict_keys(resources_per_layer)

	print(available_resources)
	print(resources_total)
	print("exiting")