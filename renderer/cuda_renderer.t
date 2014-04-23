local cuda = terralib.require("cudalib")
local renderer = {}

terra shade_pixel(params : &Params, idx : int)
   var invWidth = 1.0 / params.width
   var invHeight = 1.0 / params.height

   var cx = invWidth * ([int](idx % params.width) + 0.5)
   var cy = invHeight * ([int](idx / params.width) + 0.5)

   for i = 0, params.num_circles do
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

local kernel = cuda.make_kernel(shade_pixel)

renderer.get_image = terra(params : Params)
   var N = params.width * params.height
   kernel(&params, N)
end

return renderer