import numpy as np
import qonnx.custom_op.registry as registry
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.analysis.fpgadataflow.op_and_param_counts import aggregate_dict_keys

from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
)

from copy import deepcopy

def set_defaults(model):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	for node in model.graph.node:
		inst = registry.getCustomOp(node)
		attrs = inst.get_nodeattr_types()

		if "PE" in attrs:
			inst.set_nodeattr("PE", 1)
		
		if "SIMD" in attrs:
			inst.set_nodeattr("SIMD", 1)

def estimate_resources(model):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	res_dict = {}
	for node in model.graph.node:
		if is_hls_node(node) or is_rtl_node(node):
			inst =  registry.getCustomOp(node)
			res_dict[node.name] = inst.node_res_estimation()
	
	return res_dict

def estimate_cycles(model):
	model = model.transform(GiveUniqueNodeNames())
	model = model.transform(GiveReadableTensorNames())

	cycle_dict = {}
	for node in model.graph.node:
		if is_hls_node(node) or is_rtl_node(node):
			inst = registry.getCustomOp(node)
			cycle_dict[node.name] = int(inst.get_exp_cycles())
	
	return cycle_dict

def reduceBRAMUsage(model, resources_per_layer, available_resources, max_iters = 10):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['BRAM_18K'], reverse = True)

	resources_total = aggregate_dict_keys(resources_per_layer)

	iters = 1
	while iters < max_iters and resources_total['BRAM_18K'] > available_resources['BRAM_18K']:
		iters += 1
		
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
					if node_inst.uram_efficiency_estimation() < 0.2:
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
		
		resources_per_layer = estimate_resources(model)
		resources_total = aggregate_dict_keys(resources_per_layer)
	
	return model

def reduceDSPUsage(model, resources_per_layer, available_resources, max_iters = 10):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['DSP'], reverse = True)
	
	resources_total = aggregate_dict_keys(resources_per_layer)

	iters = 1
	while iters < max_iters and resources_total['DSP'] > available_resources['DSP']:
		iters += 1
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

		resources_per_layer = estimate_resources(model)
		resources_total = aggregate_dict_keys(resources_per_layer)
			
	return model

def reduceLUTUsage(model, resources_per_layer, available_resources, max_iters = 10):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['LUT'], reverse = True)
	
	resources_total = aggregate_dict_keys(resources_per_layer)

	iters = 1
	while iters < max_iters and resources_total['LUT'] > available_resources['LUT']:
		iters += 1
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
					if node_inst.uram_efficiency_estimation() < 0.2:
						node_inst.set_nodeattr("ram_style", "block")
					break
			elif op_type == "Thresholding_hls":
				ram_style = node_inst.get_nodeattr("ram_style")
				tmem = node_inst.calc_tmem()
				if ram_style == "distributed" and tmem > 1:
					node_inst.set_nodeattr("ram_style", "block")
					break
		
		resources_per_layer = estimate_resources(model)
		resources_total = aggregate_dict_keys(resources_per_layer)
	
			
	return model

def reduceURAMUsage(model, resources_per_layer, available_resources, max_iters = 10):
	sorted_resources_per_layer = sorted(resources_per_layer.items(), key = lambda x : x[1]['URAM'], reverse = True)
	
	resources_total = aggregate_dict_keys(resources_per_layer)
	
	iters = 1
	while iters < max_iters and resources_total['URAM'] > available_resources['URAM']:
		iters += 1
		for layer in sorted_resources_per_layer:
			name, _ = layer

			node = model.get_node_from_name(name)
			node_inst = registry.getCustomOp(node)
			op_type = node.op_type
			
			if op_type in ["MVAU_hls", "MVAU_rtl", "VVAU_hls", "VVAU_rtl"]:
				mem_mode = node_inst.get_nodeattr("mem_mode")
				if mem_mode == "internal_decoupled" and node_inst.get_nodeattr("ram_style") == "ultra":
					if node_inst.bram_efficiency_estimation() >= 0.5:
						node_inst.set_nodeattr("ram_style", "block")
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
		
		resources_per_layer = estimate_resources(model)
		resources_total = aggregate_dict_keys(resources_per_layer)
	
	return model	

def check_resources(available_resources, resources_total):
	return np.all(np.array(list(resources_total.values())) <= np.array(list(available_resources.values()))) 

def isFeasible(model, available_resources, max_iters = 10):
	resources_per_layer = estimate_resources(model)
	resources_total = aggregate_dict_keys(resources_per_layer)
	
	iters = 1
	while iters < max_iters and not check_resources(available_resources, resources_total):
		iters += 1
		if resources_total['BRAM_18K'] > available_resources['BRAM_18K']:
			model = reduceBRAMUsage(model, resources_per_layer, available_resources)
		
		if resources_total['LUT'] > available_resources['LUT']:
			model = reduceLUTUsage(model, resources_per_layer, available_resources)
		
		if resources_total['URAM'] > available_resources['URAM']:
			model = reduceURAMUsage(model, resources_per_layer, available_resources)

		if resources_total['DSP'] > available_resources['DSP']:
			model = reduceDSPUsage(model, resources_per_layer, available_resources)

		resources_per_layer = estimate_resources(model)
		resources_total = aggregate_dict_keys(resources_per_layer)

	feasible = check_resources(available_resources, resources_total)
	return model, feasible

def increase_folding(model, bottleneck_layer):
	node = model.get_node_from_name(bottleneck_layer)
	op_type = node.op_type
	node_inst = registry.getCustomOp(node)

	increased = False

	pe_ops = [
            "AddStreams_hls",
            "ChannelwiseOp_hls",
            "DuplicateStreams_hls",
            "GlobalAccPool_hls",
            "Thresholding_hls",
            "Thresholding_rtl",
        ]
	
	simd_ops = [
            "DownSampler_hls",
            "FMPadding_hls",
			"FMPadding_rtl",
            "FMPadding_Pixel_hls",
            "ConvolutionInputGenerator_hls",
            "ConvolutionInputGenerator_rtl",
        ]

	if op_type in ["MVAU_hls", "MVAU_rtl"]:
		max_simd = node_inst.get_nodeattr("MW")
		max_pe = node_inst.get_nodeattr("MH")

		cur_simd = node_inst.get_nodeattr("SIMD")
		cur_pe = node_inst.get_nodeattr("PE")

		if cur_simd < max_simd:
			for simd_val in range(cur_simd + 1, max_simd + 1):
				if (max_simd % simd_val) == 0:
					node_inst.set_nodeattr("SIMD", simd_val)
					increased = True
					break
		elif cur_pe < max_pe:
			for pe_val in range(cur_pe + 1, max_pe + 1):
				if (max_pe % pe_val) == 0:
					node_inst.set_nodeattr("PE", pe_val)
					increased = True
					break
	elif op_type in pe_ops:
		max_pe = node_inst.get_nodeattr("NumChannels")
		cur_pe = node_inst.get_nodeattr("PE")
		if cur_pe < max_pe:
			for pe_val in range(cur_pe + 1, max_pe + 1):
				if (max_pe % pe_val) == 0:
					node_inst.set_nodeattr("PE", pe_val)
					increased = True
					break
	elif op_type == "LabelSelect_hls":
		max_pe = node_inst.get_nodeattr("Labels")
		cur_pe = node_inst.get_nodeattr("PE")

		if cur_pe < max_pe:
			for pe_val in range(cur_pe + 1, max_pe + 1):
				if (max_pe % pe_val) == 0:
					node_inst.set_nodeattr("PE", pe_val)
					increased = True
					break
	elif op_type == "depthwise_op_exceptions":
		if op_type in ["VVAU_hls", "VVAU_rtl"]:
			max_simd = np.prod(node_inst.get_nodeattr("Kernel"))
			max_pe = node_inst.get_nodeattr("Channels")

			cur_simd = node_inst.get_nodeattr("SIMD")
			cur_pe = node_inst.get_nodeattr("PE")
			
			if cur_pe < max_pe:
				for pe_val in range(cur_pe + 1, max_pe + 1):
					if (max_pe % pe_val) == 0:
						node_inst.set_nodeattr("PE", pe_val)
						increased = True
						break
			elif cur_simd < max_simd:
				for simd_val in range(cur_simd + 1, max_simd + 1):
					if (max_simd % simd_val) == 0:
						node_inst.set_nodeattr("SIMD", simd_val)
						increased = True
						break

			swu_node = model.find_producer(node.input[0])
			if swu_node.op_type.startswith("ConvolutionInputGenerator"):
				swu_node_inst = registry.getCustomOp(swu_node)
				swu_node_inst.set_nodeattr("SIMD", node_inst.get_nodeattr("PE"))

				if swu_node.op_type == "ConvolutionInputGenerator_rtl":
					if op_type.startswith("VVAU") and node_inst.get_nodeattr("SIMD") > 1:
						swu_node_inst.set_nodeattr("parallel_window", 1)
					else:
						swu_node_inst.set_nodeattr("parallel_window", 0)
	elif op_type in simd_ops:
		if op_type.startswith("ConvolutionInputGenerator"):
			depthwise = node_inst.get_nodeattr("depthwise")
			if depthwise == 0:
				max_simd = node_inst.get_nodeattr("IFMChannels")
				if op_type == "ConvolutionInputGenerator_rtl":
					node_inst.set_nodeattr("parallel_window", 0)
				
				cur_simd = node_inst.get_nodeattr("SIMD")
				if cur_simd < max_simd:
					for simd_val in range(cur_simd + 1, max_simd + 1):
						if (max_simd % simd_val) == 0:
							node_inst.set_nodeattr("SIMD", simd_val)
							increased = True
							break
					if op_type == "ConvolutionInputGenerator_rtl" and node_inst.get_nodeattr("SIMD") == max_simd:
						node_inst.set_nodeattr("parallel_window", 1)
		else:
			max_simd = node_inst.get_nodeattr("NumChannels")
			cur_simd = node_inst.get_nodeattr("SIMD")
			if cur_simd < max_simd:
				for simd_val in range(cur_simd + 1, max_simd + 1):
					if (max_simd % simd_val) == 0:
						node_inst.set_nodeattr("SIMD", simd_val)
						increased = True
						break

	return model, increased 

def avg_utilization(model, available_resources):
	resources_per_layer = estimate_resources(model)
	resources_total = aggregate_dict_keys(resources_per_layer)

	avg_util = 0
	max_util = 0
	for resource in resources_total.keys():
		util = (resources_total[resource]) / available_resources[resource]
		avg_util += 1 / len(resources_total.keys()) * util

		if util > max_util:
			max_util = util
	
	return avg_util, max_util

def folding(model, available_resources, clk_period):
	set_defaults(model)
	prev_model = deepcopy(model)

	model, feasible = isFeasible(model, available_resources)

	if not feasible:
		avg_util, max_util = avg_utilization(model, available_resources)
		return model, 0.0, avg_util, max_util, False

	while feasible:
		cycles_per_layer = estimate_cycles(model)
		sorted_cycles_per_layer = sorted(cycles_per_layer.items(), key = lambda x : x[1], reverse = True)
		bottleneck_layer, latency = sorted_cycles_per_layer[0]
	
		model, increased = increase_folding(model, bottleneck_layer)
		if not increased:
			break

		prev_model = deepcopy(model)
		model, feasible = isFeasible(model, available_resources)
	
	model = deepcopy(prev_model)
	cycles_per_layer = estimate_cycles(model)
	max_cycles = max(cycles_per_layer.items(), key = lambda x : x[1])[1]
	fps = 1 / (max_cycles * clk_period) * 10**9
	avg_util, _ = avg_utilization(model, available_resources)
	return model, fps, avg_util, None, True
