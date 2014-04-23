local renderer = {}

terra clamp(n : int, min : int, max : int)
   if n > max then return max
   elseif n < min then return min
   else return n
   end
end

terra shade_pixel(i : int, cx : double, cy : double, index : int, params : Params)
   var px, py, pz = params.position[i * 3], params.position[i * 3 + 1], params.position[i * 3 + 2]
   var radius = params.radius[i]
   var color = &params.color[i * 3]

   var diffX, diffY = px - cx, py - cy
   var pixelDist = diffX * diffX + diffY * diffY

   if pixelDist > radius * radius then return end

   var alpha = 0.5
   for k = 0, 3 do
      params.data[index + k] = alpha * color[k] + (1 - alpha) * params.data[index + k]
   end

   if params.data[index + 3] + alpha < 1.0 then
      params.data[index + 3] = params.data[index + 3] + alpha
   end
end
 
renderer.get_image = terra(params : Params)
   for i = 0, params.num_circles do
      var px, py, pz = params.position[i * 3], params.position[i * 3 + 1], params.position[i * 3 + 2]
      var radius = params.radius[i]

      var minX = px - radius
      var maxX = px + radius
      var minY = py - radius
      var maxY = py + radius

      var screenMinX = clamp(minX * params.width, 0, params.width)
      var screenMaxX = clamp(maxX * params.width + 1, 0, params.width) - 1
      var screenMinY = clamp(minY * params.height, 0, params.height)
      var screenMaxY = clamp(maxY * params.height + 1, 0, params.height) - 1

      var invWidth = 1.0 / params.width
      var invHeight = 1.0 / params.height

      for pixelY = screenMinY, screenMaxY + 1 do
         for pixelX = screenMinX, screenMaxX + 1 do
            var centerX = invWidth * (pixelX + 0.5)
            var centerY = invHeight * (pixelY + 0.5)
            shade_pixel(i, centerX, centerY,
                        4 * (pixelY * params.width + pixelX), 
                        params)
         end
      end
   end
end

return renderer