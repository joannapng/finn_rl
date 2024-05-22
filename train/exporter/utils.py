import math

def uram_estimation(node):
	P = node.get_nodeattr("PE")
	Q = node.get_nodeattr("SIMD")
	wdt = node.get_weight_datatype()
	W = wdt.bitwidth()
	D_in = node.get_nodeattr("MW")
	D_out = node.get_nodeattr("MH")
	omega = (D_in * D_out) / (Q * P)
	mem_width = Q * W * P
	mmode = node.get_nodeattr("mem_mode")
	mstyle = node.get_nodeattr("ram_style")
	width_multiplier = math.ceil(mem_width / 72)
	depth_multiplier = math.ceil(omega / 4096)
	return width_multiplier * depth_multiplier

def bram_estimation(node):
	P = node.get_nodeattr("PE")
	Q = node.get_nodeattr("SIMD")
	wdt = node.get_weight_datatype()
	W = wdt.bitwidth()
	D_in = node.get_nodeattr("MW")
	D_out = node.get_nodeattr("MH")
	omega = (D_in * D_out) / (Q * P)
	mem_width = Q * W * P
	mmode = node.get_nodeattr("mem_mode")
	mstyle = node.get_nodeattr("ram_style")

	if mem_width == 1:
		return math.ceil(omega / 16384)
	elif mem_width == 2:
		return math.ceil(omega / 8192)
	elif mem_width <= 4:
		return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
	elif mem_width <= 9:
		return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 9))
	elif mem_width <= 18 or omega > 512:
		return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 18))
	else:
		return (math.ceil(omega / 512)) * (math.ceil(mem_width / 36))
        
def bram_efficiency_estimation(node):
	wdt = node.get_weight_datatype()
	W = wdt.bitwidth()
	D_in = node.get_nodeattr("MW")
	D_out = node.get_nodeattr("MH")
	bram16_est = bram_estimation(node)
	if bram16_est == 0:
		return 1
	wbits = W * D_in * D_out
	bram16_est_capacity = bram16_est * 36 * 512
	return wbits / bram16_est_capacity

def uram_efficiency_estimation(node):
	wdt = node.get_weight_datatype()
	W = wdt.bitwidth()
	D_in = node.get_nodeattr("MW")
	D_out = node.get_nodeattr("MH")
	uram_est = uram_estimation(node)
	if uram_est == 0:
		return 1
	wbits = W * D_in * D_out
	uram_est_capacity = uram_est * 72 * 4096
	return wbits / uram_est_capacity