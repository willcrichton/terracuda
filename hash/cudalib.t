terralib.includepath = terralib.includepath..";/usr/local/cuda/include"

local C = terralib.includecstring [[
#include "cuda_runtime.h"
#include <stdlib.h>
#include <stdio.h>
]]

local thread_id = cudalib.nvvm_read_ptx_sreg_tid_x
local block_dim = cudalib.nvvm_read_ptx_sreg_ntid_x
local block_id = cudalib.nvvm_read_ptx_sreg_ctaid_x

local NUM_THREADS = 64
local cuda = {}

cuda.index = terra()
   return thread_id() + block_id() * block_dim()
end

cuda.make_map_kernel = function(func)
   local ltype = func:gettype().parameters[1].type
   local kernel = terra(device_arr : &ltype)
      func(device_arr, cuda.index())
   end

   return terralib.cudacompile({kernel = kernel}).kernel
end

cuda.map = function(func)
   local mapper = cuda.make_map_kernel(func)
   local ltype = func:gettype().parameters[1].type

   return terra(host_arr : &ltype, N : int)
      var cuda_arr : &ltype
      var params : terralib.CUDAParams
      if N < NUM_THREADS then
         params = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }
      else
         params = terralib.CUDAParams { N/NUM_THREADS,1,1, NUM_THREADS,1,1, 0, nil }
      end

      C.cudaMalloc([&&opaque](&cuda_arr), sizeof(ltype) * N)
      C.cudaMemcpy(cuda_arr, host_arr, sizeof(ltype) * N, 1)
      mapper(&params, cuda_arr)
      C.cudaDeviceSynchronize()
      C.cudaMemcpy(host_arr, cuda_arr, sizeof(ltype) * N, 2)
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
      C.cudaDeviceSynchronize()
      C.cudaMemcpy(host_arr, cuda_arr, sizeof(ltype) * N, 2)
   end
end

cuda.lua_map = function(func)
   local map = cuda.map(func)
   local ltype = func:gettype().parameters[1].type -- assume first arg is array
   
   local new = terra(N : int) return C.malloc(sizeof(ltype) * N) end
   local copy = terra(a : &ltype, b : ltype, i : int) a[i] = b end
   local get = terra(a : &ltype, i : int) return a[i] end
   local free = terra(a : &ltype) C.free(a) end

   return function(list) 
      local N = #list
      local arr = new(N)
      for i = 1, N do
         copy(arr, list[i], i - 1)
      end

      map(arr, N)

      for i = 1, N do
         local temp = get(arr, i - 1)
         if ltype:isstruct() then
            for _, v in pairs(ltype.entries) do
               list[i][v.field] = temp[v.field]
            end
         else
            list[i] = temp
         end
      end

      free(arr)
   end
end

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
      func(A, cuda.index())
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
         var launch_params : terralib.CUDAParams
         if N < NUM_THREADS then
            launch_params = terralib.CUDAParams { 1,1,1, N,1,1, 0, nil }
         else
            launch_params = terralib.CUDAParams { (N/NUM_THREADS)+1,1,1, NUM_THREADS,1,1, 0, nil }
         end
         
         C.cudaMemcpy(cuda_params, &params, sizeof(s_type), 1)
         kernel(&launch_params, cuda_params)
         C.cudaDeviceSynchronize()
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

function make_array(typ, N)
   local init = terra() return [typ](cuda.alloc(sizeof(typ.type) * N)) end
   return init()
end

cuda.lua_make_kernel = function(func)
   local kernel = cuda.make_kernel(func)
   local stype = func:gettype().parameters[1].type -- assume first arg is struct pointer

   local sval = terralib.new(stype)

   return function(P, N)
      for _, entry in pairs(stype.entries) do
         if entry.type:ispointer() then -- assume it's an array
            local arr = P[entry.field]
            local tarr = make_array(entry.type, #arr)
            sval[entry.field] = tarr
            local set = terra(i : int, v : entry.type.type) tarr[i] = v end
            for i = 1, #arr do set(i - 1, arr[i]) end
         else
            sval[entry.field] = P[entry.field]
         end
      end

      kernel(sval, N)

      for _, entry in pairs(stype.entries) do
         if entry.type:ispointer() then -- assume it's an array
            local arr = P[entry.field]
            local tarr = sval[entry.field]
            local get = terra(i : int) return tarr[i] end
            for i = 1, #arr do arr[i] = get(i - 1) end
         else
            P[entry.field] = sval[entry.field]
         end
      end      
   end
end

cuda.make_struct_type = function(t)
   local st = terralib.types.newstruct("custom_struct")
   local type_map = {
      string = &int8,
      number = int, -- TODO: distinguish between floats/doubles?
      boolean = bool,
      table = &int
   }

   for k, v in pairs(t) do
      table.insert(st.entries, { field = k, type = type_map[type(v)] })
   end

   return st
end


return cuda