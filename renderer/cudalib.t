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

cuda.make_map_kernel = function(func)
   local ltype = func:gettype().parameters[1]
   local kernel = terra(device_arr : &ltype)
      var idx = thread_id() + block_id() * block_dim()
      device_arr[idx] = func(device_arr[idx])
   end

   return terralib.cudacompile({kernel = kernel}).kernel
end

cuda.map = function(func)
   local mapper = cuda.make_map_kernel(func)
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
   local mapper = cuda.make_map_kernel(func)
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

   local new = terra(N : int) return C.malloc(sizeof(ltype) * N) end
   local copy = terra(a : &ltype, b : ltype, i : int) a[i] = b end
   local launch = terra(a : &ltype, N : int) map(a, N) end
   local get = terra(a : &ltype, i : int) return a[i] end

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

local arrays = {}
cuda.alloc = terralib.cast(int -> &int, function(N)
   local init_host = terra() return [&int](C.malloc(N)) end
   local init_cuda = terra()
      var cuda_arr : &int
      C.cudaMalloc([&&opaque](&cuda_arr), N)
      return cuda_arr
   end

   local host_arr = init_host()
   local cuda_arr = init_cuda()
   table.insert(arrays, {host = host_arr, cuda = cuda_arr, size = N})

   return host_arr
end)

cuda.free = function(ptr)
   for _, v in pairs(arrays) do
       if v.host == ptr then
          local free = terra()
             C.free(v.host)
             C.cudaFree(v.cuda)
          end
          free()
       end
   end
end

cuda.alloc_device = terra(N : int) : &int
   var cuda_arr : &int
   C.cudaMalloc([&&opaque](&cuda_arr), N)
   return cuda_arr
end

cuda.device_free = terra(ptr : &opaque)
   C.cudaFree(ptr)
end

--[[terra wtf()
   var x : &int
   cuda.make_array(x, 10)
end

wtf()

--[[
allocated
]]--

cuda.make_kernel = function(func)
   -- assume that our function takes one argument which is a ptr to a struct
   local sptr_type = func:gettype().parameters[1]

   if not (sptr_type:ispointertostruct())  then 
      error('Type must be a struct') 
   end

   local s_type = sptr_type.type
   local params = terralib.new(s_type)
   local init = terra() 
      var cuda_params : sptr_type
      C.cudaMalloc([&&opaque](&cuda_params), sizeof(s_type))
      return cuda_params
   end
   local cuda_params = init()

   local func_wrapper = terra(A : sptr_type) 
      func(A, thread_id() + block_id() * block_dim())
   end

   local kernel = terralib.cudacompile({kernel = func_wrapper}).kernel
      
   return function(p, N)
      for _, entry in pairs(s_type.entries) do
         if entry.type:ispointer() then
            local found = false
            for _, v in pairs(arrays) do
               if v.host == p[entry.field] then
                  params[entry.field] = terralib.cast(entry.type, v.cuda)
                  local copy = terra() C.cudaMemcpy(v.cuda, v.host, v.size, 1) end
                  copy()
                  found = true
                  break
               end
            end

            if not found then params[entry.field] = p[entry.field] end
         else
            params[entry.field] = p[entry.field]
         end
      end
         
      local launch = terra()
         var launch_params = terralib.CUDAParams { N/NUM_THREADS,1,1, NUM_THREADS,1,1, 0, nil }
         C.cudaMemcpy(cuda_params, &params, sizeof(s_type), 1)
         kernel(&launch_params, cuda_params)
         C.cudaMemcpy(&params, cuda_params, sizeof(s_type), 2)
      end

      launch()

      for _, entry in pairs(s_type.entries) do
         if entry.type:ispointer() then
            for _, v in pairs(arrays) do
               if v.host == p[entry.field] then
                  local copy = terra() C.cudaMemcpy(v.host, v.cuda, v.size, 2) end
                  copy()
               end
            end
         else
            p[entry.field] = params[entry.field]
         end
      end
   end      
end

struct A { p : int[64], q: &int, r: int }
terra mah_kernel(x : &A, idx : int)
   x.p[idx] = idx
   x.q[idx] = idx + 1
end

--[[local some_func = cuda.make_kernel(mah_kernel)

terra asdf()
   var huh : A
   huh.q = cuda.alloc(sizeof(int) * 64)
   
   some_func(&huh, 64)

   for i = 0, 3 do
      print(huh.q[i])
   end
end

]]--

return cuda