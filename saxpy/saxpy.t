local N = 10000000
local P = {A = {}, B = {}, C = {}}

for i = 1, N do
   P.A[i] = i
   P.B[i] = N - i
   P.C[i] = 0
end

local cuda = terralib.require("../cudalib")
local ptype = cuda.make_struct_type(P)
local kernel = cuda.lua_make_kernel(
terra(p : &ptype, idx : int)
   p.C[idx] = p.A[idx] + p.B[idx]
end)

kernel(P, N)