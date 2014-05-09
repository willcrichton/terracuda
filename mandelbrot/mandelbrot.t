local cuda = terralib.require("cudalib")
local P = {image = {}, N = 1024}
for i = 1, P.N * P.N * 3 do P.image[i] = 0 end

local stype = cuda.make_struct_type(P)
local render = cuda.lua_make_kernel(
terra(p : &stype, idx : int)
   if idx >= P.N * P.N then return end

   var x_dim : float, y_dim : float = idx % p.N, idx / p.N
   var x_origin, y_origin = (x_dim / p.N) * 3.25 - 2.0, (y_dim / p.N) * 2.5 - 1.25
   
   var x, y = 0.0, 0.0
   var scale = 8
   var iteration, max_iteration = 0, 256 * scale

   while x * x + y * y <= 4 and iteration < max_iteration do
      var xtemp = x * x - y * y + x_origin
      y = 2 * x * y + y_origin
      x = xtemp
      iteration = iteration + 1
   end
   
   if iteration ~= max_iteration then
      p.image[idx * 3] = iteration / scale
      p.image[idx * 3 + 1] = iteration / scale
      p.image[idx * 3 + 2] = iteration / scale
   else
      p.image[idx * 3] = 0
      p.image[idx * 3 + 1] = 0
      p.image[idx * 3 + 2] = 0
   end
end)

render(P, P.N * P.N)
