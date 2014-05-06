local cuda = terralib.require("cudalib")

terra A(i : double, j : double) : double
  i = i + 1
  j = j + 1
  var ij : double = i + j-1
  return 1.0 / (ij * (ij-1) * 0.5 + i)
end

struct Params {
   x : &double,
   y : &double,
   N : int }

local Av = cuda.make_kernel(
terra(p : &Params, i : int)
   var a = 0
   for j = 0, p.N do a = a + p.x[j] * A(i, j) end
   p.y[0] = p.y[0] + i
end)

local Atv = cuda.make_kernel(
terra(p : &Params, i : int)
   var a = 0
   for j= 0, p.N do a = a + p.x[j] * A(j, i) end
   p.y[i] = a
end)

local C = terralib.includec('stdio.h')
terra AtAv(x : &double, y : &double, t : &double, N : int)
   var p1, p2 = Params { x, t, N }, Params { t, y, N }

   Av(&p1, N)

   for i = 0, N do C.printf("%f\n", t[i]) end
   C.printf("\n")

   Atv(&p2, N)
end

local N = tonumber(arg and arg[1]) or 100
terra main()
   var u = [&double](cuda.alloc(sizeof(double) * N))
   var v = [&double](cuda.alloc(sizeof(double) * N))
   var t = [&double](cuda.alloc(sizeof(double) * N))

   for i = 0, N do u[i] = 1.0 end

   for i = 0, 1 do
      AtAv(u, v, t, N)
      AtAv(v, u, t, N)
   end

   for i = 0, N do
      --C.printf("%f %f\n", u[i], v[i])
   end

--[[

   var vBv, vv = 0.0, 0.0
   for i = 0, N do
      var ui, vi = u[i], v[i]
      vBv = vBv + ui * vi
      vv = vv + vi * vi
   end

   C.printf("%f %f\n", vBv, vv)]]
end

main()
--io.write(string.format("%0.9f\n", math.sqrt(main())))