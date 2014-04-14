local renderer = {}

local thread_id = cudalib.nvvm_read_ptx_sreg_tid_x
local block_dim = cudalib.nvvm_read_ptx_sreg_ntid_x
local block_id = cudalib.nvvm_read_ptx_sreg_ctaid_x

terralib.includepath = terralib.includepath .. ";/usr/local/cuda/include"

local C = terralib.includecstring [[
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
]]

local NUM_THREADS = 64

local cuda_data = global(&double)
local cuda_params = global(&Params)
local init_complete = global(bool, false)

terra init(params : Params)
   var radius : &double
   var position : &double
   var color : &double

   C.cudaMalloc([&&opaque](&radius), sizeof(double) * params.num_circles)
   C.cudaMalloc([&&opaque](&position), sizeof(double) * params.num_circles * 3)
   C.cudaMalloc([&&opaque](&color), sizeof(double) * params.num_circles * 3)
   C.cudaMalloc([&&opaque](&cuda_data), sizeof(double) * params.width * params.height * 4)

   C.cudaMemcpy(radius, params.radius, sizeof(double) * params.num_circles, 1)
   C.cudaMemcpy(position, params.position, sizeof(double) * params.num_circles * 3, 1)
   C.cudaMemcpy(color, params.color, sizeof(double) * params.num_circles * 3, 1)

   var p : Params
   p.radius = radius
   p.position = position
   p.color = color
   p.num_circles = params.num_circles
   p.width = params.width
   p.height = params.height
   p.data = cuda_data

   C.cudaMalloc([&&opaque](&cuda_params), sizeof(Params))
   C.cudaMemcpy(cuda_params, &p, sizeof(Params), 1)
end

terra shade_pixel()
   var idx : int = thread_id() + block_id() * block_dim()
   
   for i = 0, 4 do
      cuda_params.data[idx * 4 + i] = 1.0
   end
end

local kernel = terralib.cudacompile({ shade_pixel = shade_pixel })

renderer.get_image = terra(params : Params)
   var N = params.width * params.height

   if not init_complete then
      init(params)
      init_complete = true
   end

   C.cudaMemcpy(cuda_data, params.data, sizeof(double) * N * 4, 1)
   
   var launch = terralib.CUDAParams { N/NUM_THREADS,1,1, NUM_THREADS,1,1, 0, nil }
   kernel.shade_pixel(&launch)
   
   C.cudaMemcpy(params.data, cuda_data, sizeof(double) * N * 4, 2)
end

return renderer