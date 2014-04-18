terralib.includepath = terralib.includepath..";/usr/local/cuda/include"

local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]

local thread_id = cudalib.nvvm_read_ptx_sreg_tid_x
local block_dim = cudalib.nvvm_read_ptx_sreg_ntid_x
local block_id = cudalib.nvvm_read_ptx_sreg_ctaid_x
local __syncthreads = cudalib.cuda_syncthreads
local NUM_THREADS = 64

local cuda = {}

cuda.make_kernel = function(func)
   local ltype = func:gettype().parameters[1]
   local kernel = terra(device_arr : &ltype)
      var idx = thread_id() + block_id() * block_dim()
      device_arr[idx] = func(device_arr[idx])
   end

   return terralib.cudacompile({kernel = kernel}).kernel
end

cuda.map = function(func)
   local mapper = cuda.make_kernel(func)
   local ltype = func:gettype().parameters[1]

   return terra(host_arr : &ltype, N : int)
      var cuda_arr : &ltype
      var params = terralib.CUDAParams { N/NUM_THREADS,1,1, NUM_THREADS,1,1, 0, nil }
      
      C.cudaMalloc([&&opaque](&cuda_arr), sizeof(ltype) * N)
      C.cudaMemcpy(cuda_arr, host_arr, sizeof(ltype) * N, 1)
      mapper(&params, cuda_arr)
      C.cudaMemcpy(host_arr, cuda_arr, sizeof(ltype) * N, 2)
      C.cudaDeviceSynchronize()
   end
end

cuda.fixed_map = function(func, host_arr, N)
   local mapper = cuda.make_kernel(func)
   local ltype = func:gettype().parameters[1]

   local init = terra()
      var cuda_arr : &ltype
      C.cudaMalloc([&&opaque](&cuda_arr), sizeof(ltype) * N)
      return cuda_arr
   end

   local cuda_arr = init()
   return terra()
      var params = terralib.CUDAParams { N/NUM_THREADS,1,1, NUM_THREADS,1,1, 0, nil }
      C.cudaMemcpy(cuda_arr, host_arr, sizeof(ltype) * N, 1)
      mapper(&params, cuda_arr)
      C.cudaMemcpy(host_arr, cuda_arr, sizeof(ltype) * N, 2)
      C.cudaDeviceSynchronize()
   end
end

cuda.lua_map = function(func)
   local map = cuda.map(func)
   local ltype = func:gettype().parameters[1]

   local new = terra(N : int)
      return C.malloc(sizeof(ltype) * N)
   end

   local copy = terra(a : &ltype, b : ltype, i : int)
      a[i] = b
   end

   local launch = terra(a : &ltype, N : int)
      map(a, N)
   end

   local get = terra(a : &ltype, i : int)
      return a[i]
   end

   return function(list) 
      local N = #list
      local arr = new(N)
      for i = 1, N do
         copy(arr, list[i], i - 1)
      end

      launch(arr, N)

      for i = 1, N do
         list[i] = get(arr, i - 1)
      end
   end
end

cuda.array = function(ltype, N)
   return C.cudaMalloc(sizeof(ltype) * N)
end

local arr = global(&int)
terra mapper(n : int) : int
   return n * 2
end

local N = 64
local f = cuda.fixed_map(mapper, arr, N)

terra foo()
   arr = [&int](C.malloc(sizeof(int) * N))
   for i = 0, N do
      arr[i] = i
   end

   f()

   for i = 0, N do
      print(arr[i])
   end
end

-- foo()

struct A { p : &int, q: int }
terra mah_kernel(x : A)
   -- do somethin
end

cuda.kerneler = function(func)
   local typ = func:gettype().parameters[1]
   --for k, v in pairs(typ.entries) do
      --print(v.type:ispointer())
      --end
   
   local wot = terra()

   end
end

cuda.kerneler(mah_kernel)

return cuda