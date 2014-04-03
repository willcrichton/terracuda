local renderer = {}

function clamp(n, min, max)
   if n > max then n = max
   elseif n < min then n = min
   end
   return n
end

function shade_pixel(circle, center, index, params)
   local px, py, pz = unpack(circle.position)
   local cx, cy = unpack(center)
   local diffX, diffY = px - cx, py - cy
   local pixelDist = diffX * diffX + diffY * diffY

   if pixelDist > circle.radius * circle.radius then return end

   local alpha = 0.5
   for i = 1, 3 do
      params.data[index + i - 1] = alpha * circle.color[i] + (1 - alpha) * params.data[index + i - 1]
   end

   params.data[index + 3] = params.data[index + 3] + alpha;
end

renderer.get_image = function(params)
   for index, circle in pairs(params.circles) do
      local px, py, pz = unpack(circle.position)

      local minX = px - circle.radius
      local maxX = px + circle.radius
      local minY = py - circle.radius
      local maxY = py + circle.radius

      local screenMinX = clamp(minX * params.width, 0, params.width)
      local screenMaxX = clamp(maxX * params.width + 1, 0, params.width) - 1
      local screenMinY = clamp(minY * params.height, 0, params.height)
      local screenMaxY = clamp(maxY * params.height + 1, 0, params.height) - 1

      local invWidth = 1.0 / params.width
      local invHeight = 1.0 / params.height

      for pixelY = screenMinY, screenMaxY do
         for pixelX = screenMinX, screenMaxX do
            local centerX = invWidth * (pixelX + 0.5)
            local centerY = invHeight * (pixelY + 0.5)
            local color = shade_pixel(circle, 
                                      {centerX, centerY}, 
                                      4 * (pixelY * params.width + pixelX), 
                                      params)
         end
      end
   end
end

return renderer