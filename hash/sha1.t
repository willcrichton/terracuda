local cuda = terralib.require("cudalib")
local C = terralib.includec('stdio.h')
local sha1 = {}

terra rot_left(x : int, n : int)
   return (x << n) or (x >> (32-n))
end

terra f(t : int, b : int, c : int , d : int)
   if t <= 19 then
      return (b and c) or ((not b) and d)
   elseif t <= 39 then
      return b ^ c ^ d
   elseif t <= 59 then
      return (b and c) or (b and d) or (c and d)
   else
      return b ^ c ^ d
   end
end

terra k(t : int)
   if t <= 19 then
      return 0x5A827999
   elseif t <= 39 then
      return 0x6ED9EBA1
   elseif t <= 59 then
      return 0x8F1BBCDC
   else
      return 0xCA62C1D6
   end
end

-- computes the sha1 hash of a single message
terra sha1_hash(params : &Params, idx : int)
   --[[

   -- just to be sure
   if idx >= params.num_msgs then
      return 0
   end
   var mask : uint = 0x0000000F
   var H0 : uint = 0x67452301
   var H1 : uint = 0xEFCDAB89
   var H2 : uint = 0x98BADCFE
   var H3 : uint = 0x10325476
   var H4 : uint = 0xC3D2E1F0
   var W : &uint = &params.msgs[16*idx]
   -- Process M(i)
   var A, B, C, D, E = H0, H1, H2, H3, H4
   for t=0,80 do
      var s : int = t and mask
      if t >= 16 then
          W[s] = rot_left(W[(s + 13) and mask] ^ W[(s + 8) and mask] ^
                            W[(s + 2) and mask] ^ W[s], 1)
      end
      var tmp = rot_left(A, 5) + f(t, B, C, D) + E + W[s] + k(t)
      E = D
      D = C
      C = rot_left(B, 30)
      B = A
      A = tmp
   end
   H0,H1,H2,H3,H4 = H0 + A, H1 + B, H2 + C, H3 + D, H4 + E
   params.results[idx*5 + 0] = H0
   params.results[idx*5 + 1] = H1
   params.results[idx*5 + 2] = H2
   params.results[idx*5 + 3] = H3
   params.results[idx*5 + 4] = H4
   --]]
end

local sha_kernel = cuda.make_kernel(sha1_hash)

sha1.hash = terra(params : Params)
   sha_kernel(&params, params.num_msgs)
end

return sha1
