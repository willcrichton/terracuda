struct A { arr : &int }
some_global = global(A)

local C = terralib.includec('stdlib.h')
terra foo()
   some_global.arr = [&int](C.malloc(sizeof(int) * 3))
   some_global.arr[0] = 1
   print(some_global.arr)
   print(some_global.arr[0])
end

terra bar()
   print(some_global.arr)
   print(some_global.arr[0])
end

terra baz()
   print(some_global.arr[0])
end

foo()
bar()
baz()
