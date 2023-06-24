import Metal, Cocoa, libdispatch
from typing import List, Any
from tinygrad.helpers import prod, getenv, DEBUG, DType, dtypes
from tinygrad.runtime.lib import RawBufferMapped
import tinygrad.helpers as helpers
import numpy as np



mtl_buffers_in_flight: List[Any] = []
device = Metal.MTLCreateSystemDefaultDevice()
mtl_queue = device.newCommandQueue()

def synchronize():
    for cbuf in mtl_buffers_in_flight: cbuf.waitUntilCompleted()
    mtl_buffers_in_flight.clear()

def unwrap(x):
  ret, err = x
  assert err is None, str(err)
  return ret



shader = """
#include <metal_stdlib>
using namespace metal;
kernel void E_4(device char* data0, const device half* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    //float4 val1_0 = (float4)(((device half4*)data1)[0]);
    //data0[0] = val1_0.x;
    //data0[1] = val1_0.y;
    //data0[2] = val1_0.z;
    //data0[3] = val1_0.w;
    data0[0] = data1[0];
    data0[1] = data1[1];
    data0[2] = data1[2];
    data0[3] = data1[3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (0 == 0) {
  } /* local */
 /* global */
}
"""
shader = """
#include <metal_stdlib>
using namespace metal;
// with float4 disabled
kernel void E_4(device char* data0, const device half* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    half val1_0 = data1[0];
    half val1_1 = data1[1];
    half val1_2 = data1[2];
    half val1_3 = data1[3];
    //threadgroup_barrier(mem_flags::mem_threadgroup);
    data0[0] = val1_0;
    data0[1] = val1_1;
    data0[2] = val1_2;
    data0[3] = val1_3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (0 == 0) {
  } /* local */
 /* global */
}
"""

shader = """
#include <metal_stdlib>
using namespace metal;
kernel void E_4(device char* data0, const device half* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    float4 val1_0 = (float4)(((device half4*)data1)[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data0[0] = val1_0.x;
    data0[1] = val1_0.y;
    data0[2] = val1_0.z;
    data0[3] = val1_0.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (0 == 0) {
  } /* local */
 /* global */
}
"""

shader = """
#include <metal_stdlib>
using namespace metal;
kernel void E_4(device char* data0, const device half* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    float4 val1_0 = (float4)(((device half4*)data1)[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    data0[0] = val1_0.x;
    data0[1] = val1_0.y;
    data0[2] = val1_0.z;
    data0[3] = val1_0.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (0 == 0) {
  } /* local */
 /* global */
}
"""

shader = """
#include <metal_stdlib>
using namespace metal;
kernel void E_4(device char* data0, const device half* data1, uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    float4 val1_0 = (float4)(((device half4*)data1)[0]);
    //simdgroup_barrier(mem_flags::mem_threadgroup);
    data0[0] = val1_0.x + data0[1];
    data0[1] = val1_0.y + val1_0.w;
    data0[2] = val1_0.z;
    data0[3] = val1_0.w + val1_0.y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (0 == 0) {
  } /* local */
 /* global */
}
"""

options = Metal.MTLCompileOptions.alloc().init()
library = unwrap(device.newLibraryWithSource_options_error_(shader, options, None))
fxn = library.newFunctionWithName_('E_4')

pipeline_state = unwrap(device.newComputePipelineStateWithFunction_error_(fxn, None))
print(pipeline_state.threadExecutionWidth())

# disasemble
# arc = unwrap(device.newBinaryArchiveWithDescriptor_error_(Metal.MTLBinaryArchiveDescriptor.alloc().init(), None))
# desc = Metal.MTLComputePipelineDescriptor.alloc().init()
# desc.setComputeFunction_(fxn)
# unwrap(arc.addComputePipelineFunctionsWithDescriptor_error_(desc, None))
# unwrap(arc.serializeToURL_error_(Cocoa.NSURL.URLWithString_("file:///tmp/shader.bin"), None))
# # clone https://github.com/dougallj/applegpu.git in tinygrad/disassemblers
# os.system(f"cd {pathlib.Path(__file__).parent.parent.parent}/disassemblers/applegpu && python3 compiler_explorer.py /tmp/shader.bin")

class RawMetalBuffer(RawBufferMapped):
  def __init__(self, size:int, dtype:DType):
    assert dtype != dtypes.float64, "metal doesn't support float64"
    super().__init__(size, dtype, device.newBufferWithLength_options_(size*dtype.itemsize, Metal.MTLResourceStorageModeShared))
  def __del__(self):
    self._buf.release()
    super().__del__()
  def _buffer(self):
    synchronize()
    return self._buf.contents().as_buffer(self._buf.length())


buf1 = RawMetalBuffer(4, helpers.dtypes.int8)
buf2 = RawMetalBuffer(4, helpers.dtypes.float16)

buf1._copyin(np.array([100, 100, 100, 100], dtype=np.int8))
buf2._copyin(np.array([1, 2, 3, 4], dtype=np.float16))

bufs = (buf1, buf2)



command_buffer = mtl_queue.commandBuffer()
encoder = command_buffer.computeCommandEncoder()
encoder.setComputePipelineState_(pipeline_state)
for i,a in enumerate(bufs): encoder.setBuffer_offset_atIndex_(a._buf, 0, i)
encoder.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSize(*[4, 1, 1]), Metal.MTLSize(*[4, 1, 1]))
encoder.endEncoding()
command_buffer.commit()

command_buffer.waitUntilCompleted()

print(buf1.toCPU())
print(buf2.toCPU())

