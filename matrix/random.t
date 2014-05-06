local N = tonumber(arg[1]) or 1000

print(N)
for k = 1, 2 do
   for i = 1, N do
      local str = ""
      for j = 1, N do
         str = str .. string.format("%d", math.random() * 10) .. " "
      end
      print(str)
   end
end