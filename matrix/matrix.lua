local A, B, C = {}, {}, {}

local N = nil
while true do
   local line = io.read()
   if line == nil then break end

   if N == nil then N = tonumber(line)
   else
      local row = {}
      for i in string.gmatch(line, "%S+") do 
         table.insert(row, tonumber(i))
      end
      
      table.insert(#A < N and A or B, row)
   end
end

for i = 1, N do
   C[i] = {}

   for j = 1, N do
      local sum = 0
      for k = 1, N do
         sum = sum + A[i][k] * B[k][j]
      end

      C[i][j] = sum
   end
end

-- todo: output