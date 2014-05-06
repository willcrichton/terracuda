local N = 10000000
local P = {A = {}, B = {}, C = {}}

for i = 1, N do
   P.A[i] = i
   P.B[i] = N - i
   P.C[i] = 0
end

for i = 1, N do
   P.C[i] = P.A[i] + P.B[i]
end