local cuda = terralib.require("cudalib")

local q = {}
local N = 100000

for i = 1, N do q[i] = i end

terra do_work(x : int) : int
   var y : int = 0
   for i = 0, 10000 do
      if i % 3 == 0 or i % 5 == 0 then
         y = y + 1
      end
   end
   return x + y
end

function parallel()
   local kernel = cuda.lua_map(do_work)
   kernel(q)
end

function serial()
   local qq = global(int[N])
   qq:set(q)
   
   local do_map = terra()
      for i = 0, N do
         qq[i] = do_work(qq[i])
      end
   end

   do_map()

   local get_i = terra(i : int) return qq[i - 1] end
   for i = 1, N do
      q[i] = get_i(i)
   end
end

local start = os.clock()
serial()
print("Terra time: ", os.clock() - start)

start = os.clock()
parallel()
print("CUDA time: ", os.clock() - start)

