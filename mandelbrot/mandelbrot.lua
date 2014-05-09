local P = {image = {}, N = 1024}

for y_dim = 1, P.N do
   for x_dim = 1, P.N do
      local x_origin, y_origin = (x_dim / P.N) * 3.25 - 2.0, (y_dim / P.N) * 2.5 - 1.25
      local idx = (y_dim * P.N + x_dim) * 3
      local x, y = 0.0, 0.0
      local scale = 8
      local iteration, max_iteration = 0, 256 * scale

      while x * x + y * y <= 4 and iteration < max_iteration do
         local xtemp = x * x - y * y + x_origin
         y = 2 * x * y + y_origin
         x = xtemp
         iteration = iteration + 1
      end
      
      if iteration ~= max_iteration then
         local val = math.floor(iteration / scale)
         P.image[idx * 3] = val
         P.image[idx * 3 + 1] = val
         P.image[idx * 3 + 2] = val
      else
         P.image[idx * 3] = 0
         P.image[idx * 3 + 1] = 0
         P.image[idx * 3 + 2] = 0
      end
   end
end
