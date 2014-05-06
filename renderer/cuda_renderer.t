local cuda = terralib.require("cudalib")
local renderer = {}

local floor = cudalib.nvvm_floor_d
local to_int = cudalib.nvvm_d2i_rm
local to_float = cudalib.nvvm_i2f_rm

terra shade_pixel(params : &Params, idx : int)

   var cx = (idx % params.width + 0.5) / params.width
   var cy = (idx / params.width + 0.5) / params.height

   -- find corresponding leaf in circle tree for our pixel
   var x = floor(cx / params.tree_threshold) * params.tree_threshold -- cx / params.tree_threshold
   var y = floor(cy / params.tree_threshold) * params.tree_threshold
   var tid = to_int((x + y * params.tree_dim) * params.tree_dim)
   var circles = &params.tree[tid * params.num_circles]

   -- limit containment test to only circles inside the circle tree region
   for c = 0, params.node_size[tid] do
      var i = circles[c]

      var position = &params.position[i * 3]
      var radius = params.radius[i]
      var color = &params.color[i * 3]
      
      var diffX = position[0] - cx
      var diffY = position[1] - cy
      var pixelDist = diffX * diffX + diffY * diffY
      
      if (pixelDist <= radius * radius) then
         for j = 0, 4 do
            var k = idx * 4 + j
            if j < 3 then
               params.data[k] = 0.5 * color[j] + 0.5 * params.data[k]
            elseif params.data[k] + 0.5 <= 1.0 then
               params.data[k] = 0.5 + params.data[k]
            end
         end
      end
   end
end

terra clamp(x : double, min : double, max : double)
   if x < min then return min
   elseif x > max then return max
   else return x end
end

terra build_tree(params : &Params, idx : int)
   if idx >= params.tree_dim * params.tree_dim then return end

   var box = array((idx % params.tree_dim) / to_float(params.tree_dim),
                   (idx / params.tree_dim) / to_float(params.tree_dim),
                   params.tree_threshold, params.tree_threshold)

   var counter : int = 0
   for i = 0, params.num_circles do
      var radius = params.radius[i]
      var circleX = params.position[i * 3]
      var circleY = params.position[i * 3 + 1]
      
      var closeX = clamp(circleX, box[0], box[0] + box[2]) 
      var closeY = clamp(circleY, box[1], box[1] + box[3]) 

      var distX = circleX - closeX
      var distY = circleY - closeY

      if (distX * distX + distY * distY) < radius * radius then
         params.tree[idx * params.num_circles + counter] = i
         counter = counter + 1
      end
   end

   params.node_size[idx] = counter
end

local pixel_kernel = cuda.make_kernel(shade_pixel)
local tree_kernel = cuda.make_kernel(build_tree)

local T = terralib.includec('sys/time.h')
terra time() : float
   var tv : T.timeval, tz : T.timezone
   T.gettimeofday(&tv, &tz)
   return tv.tv_sec + tv.tv_usec / 1000000.0
end

local C = terralib.includec('stdio.h')
renderer.get_image = terra(params : Params)
   var tv : T.timeval, tz : T.timezone
   T.gettimeofday(&tv, &tz)
   var start = tv.tv_sec + tv.tv_usec / 1000000.0

   tree_kernel(&params, params.tree_dim * params.tree_dim)
   
   T.gettimeofday(&tv, &tz)
   var after = tv.tv_sec + tv.tv_usec / 1000000.0
   C.printf("Tree time: %f\n", after - start);

   pixel_kernel(&params, params.width * params.height)

   T.gettimeofday(&tv, &tz)
   C.printf("Pixel time: %f\n", ( tv.tv_sec + tv.tv_usec / 1000000.0) - after);
end

return renderer