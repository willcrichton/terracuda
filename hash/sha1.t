local cuda = terralib.require("cudalib")
local C = terralib.includec('stdio.h')
local sha1 = {}

terra rot_left(x : int, n : int)
   return (x << n) or (x >> (32-n))
end

terra f(t : int, b : int, c : int , d : int)
   assert(0 <= t and t < 80)
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
   assert(0 <= t and t < 80)
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
   C.printf("Hashing index: %d\n", idx)
   -- just to be sure
   if idx >= params.num_msgs then
      return 0
   end

   -- number of blocks for this message
   var num_blocks : int = params.block_lengths[idx]
   -- starting word for this set of blocks for this message
   var block_idx : &int = &params.words[params.start_block[idx]]
   -- every block has its own 160 bit result message digest

   var mask : int = 0x0000000F
   var H0 : int = 0x67452301
   var H1 : int = 0xEFCDAB89
   var H2 : int = 0x98BADCFE
   var H3 : int = 0x10325476
   var H4 : int = 0xC3D2E1F0

   var W : &int = block_idx
   -- Process M(i)
   for i=0,num_blocks do
      var A, B, C, D, E = H0, H1, H2, H3, H4
      for t=0,79 do
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
   end
   C.printf("%d %d %d %d %d\n", H0, H1, H2, H3, H4)
end


local sha_kernel = cuda.make_kernel(sha1_hash)

sha1.hash = terra(params : Params)
   sha_kernel(&params, params.num_msgs)
end

return sha1
