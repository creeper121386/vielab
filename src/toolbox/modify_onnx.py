import sys

import onnx

# from globalenv import console

# 一般先加载模型
# print = console.log
onnx_path = sys.argv[1]
out_onnx_path = sys.argv[2]
onnx_model = onnx.load(onnx_path)
graph = onnx_model.graph
node = graph.node

# 查看所有node结构，对应这onnx文件查看，onnx用netron可视化比较好。
# for i in range(len(node)):
#     node_str = str(node[i])
#     if len(node_str) > 1024:
#         node_str = 'Node content too long, skip print.'
#     print(f'[ Node{i} ] {node_str}')
#
# # 查看输入输出
# for n in graph.input:
#     print('\n[ Inputs ] ')
#     print(n)
# for n in graph.output:
#     print('\n[ Outputs ] ')
#     print(n)

# >>>>> 修 改 结 构 <<<<<

# 删除节点
# 注意，所有node是有一定排序的，删除一个节点后，其后续节点的排序会改变，
# 所有删除节点时一定要先删除序号较后的节点，以免排序改变
graph.node.remove(node[76])
graph.node.remove(node[65])
graph.node.remove(node[64])

# 删除输入
# for n in graph.input:
#     if n.name == 'lr1':
#         graph.input.remove(n)

# 指定node的输入输出
# node[77].input[0] = 'lr'
# node[47].output[0] = 'sr'
# node[50].output[0] = '29'  # 节点序号

# 改变输入输出维度
# graph.input[0].type.tensor_type.shape.dim[0].dim_value = 1
# graph.input[0].type.tensor_type.shape.dim[1].dim_value = 48
# graph.input[0].type.tensor_type.shape.dim[2].dim_value = 180
# graph.input[0].type.tensor_type.shape.dim[3].dim_value = 320

# 　新建节点，插入模型中
new_node = onnx.helper.make_node(
    "ReshapeGuidemapAndGridSample",
    inputs=['145', '157'],
    outputs=['158'],
)
graph.node.insert(999, new_node)

# node[77].input = '158'

# 保存onnx
# onnx.checker.check_model(onnx_model)  # 检查模型是否有异常，有时可以跳过
onnx.save(onnx_model, out_onnx_path)
