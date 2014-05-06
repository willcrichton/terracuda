local P = {A = {}, B = {}, C = {}, N = nil}

while true do
   local line = io.read()
   if line == nil then break end

   if P.N == nil then P.N = tonumber(line)
   else
      for i in string.gmatch(line, "%S+") do 
         table.insert(#P.A < P.N * P.N and P.A or P.B, tonumber(i))
      end
   end
end

for i = 1, P.N * P.N do P.C[i] = 0 end

local cuda = terralib.require("cudalib")
local stype = cuda.make_struct_type(P)
local kernel = cuda.lua_make_kernel(
terra(p : &stype, idx : int)
   if idx > p.N * p.N then return end

   var i, j, sum = idx / p.N, idx % p.N, 0
   for k = 0, p.N do
      sum = sum + p.A[i * p.N + k] * p.B[k * p.N + j]
   end

   p.C[idx] = sum
end)

kernel(P, P.N * P.N)