local char,rep = string.char, string.rep
local modf = math.modf

terra bor(a : int, b : int)
   return a or b
end

terra band(a : int, b : int)
   return a and b
end

terra bxor(a : int, b : int)
   return a ^ b
end

terra bnot(a : int)
   return not a
end
terra bshiftl(a : int, n : int)
   return a << n
end

terra bshiftr(a : int, n : int)
   return a >> n
end

-- given a word in 4 individual bytes, convert to a single value
function bytes_to_word(a, b, c, d)
   return a*0x1000000 + b*0x10000 + c*0x100 + d
end

-- Lua 5.1 doesn't support bitwise operators, really lua? really?
-- use terra for simplicity, I don't want to deal with individual
--   individual bits -> booleans

-- rotate a word by n bits (to the left)
function rotate_word(w, n)
   return bor(bshiftl(w, n),
               bshifr(w, (32-n)))
end

-- defined by sha1 algorithm
function f(t, b, c, d)
   if 0 <= t and t <= 19 then
      return bor(band(b, c),
                  band(bnot(b), d))
   elseif 20 <= t and t <= 39 then
      return bxor(b, bxor(c, d))
   elseif 40 <= t and t <= 59 then
      return bor(band(b, c),
                  bor(band(b, d), band(c, d)))
   elseif 60 <= t and t <= 79 then
      return bxor(b, bxor(c, d))
   else
      error("invalid t (f)")
   end
end

-- defined by sha1 algorithm
function k(t)
   if 0 <= t and t <= 19 then
      return 0x5A827999
   elseif 20 <= t and t <= 39 then
      return 0x6ED9EBA1
   elseif 40 <= t and t <= 59 then
      return 0x8F1BBCDC
   elseif 60 <= t and t <= 79 then
      return 0xCA62C1D6
   else
      error("invalid t (k)")
   end
end

function msg_to_blocks(msg)
   local msg_len_bits = #msg * 8 -- message length in bits

   -- +9 includes the first (10000000) byte, and the last 8 bytes of length
   local msg_len_bytes = #msg + 9 -- message length in bytes

   local append_one = char(0x80) -- 1 followed by 7 0s

   local byte1, rem1 = modf(msg_len_bits / 0x01000000)
   local byte2, rem2 = modf(0x01000000 * rem1 / 0x00010000)
   local byte3, rem3 = modf(0x00010000 * rem2 / 0x00000100)
   local byte4 = 0x00000100 * rem3

   -- these are the last 64 bits (assuming len(msg) < 2^32 - 1)
   local append_length = char(0) .. char(0) .. char(0) .. char(0) ..
      char(byte1) .. char(byte2) .. char(byte3) .. char(byte4)

   -- append all of the zeroes inbetween
   -- since we only have char(0x00) at our disposal, use mod 64 instead of 512
   -- this is the number of 0x00 bytes to add inbetween
   local mod64 = msg_len_bytes % 64
   local append_zero = ""
   if mod64 > 0 then append_zero = rep(char(0x00), 64-mod64) end

   -- create the 'block', 512 bits of pure sweetness, using the appends we made
   local msg_block = msg .. append_one .. append_zero .. append_length
   -- be sure the message mod 512 (in bits) is 0. Need it to be a
   -- multiple of 512. pls.
   assert(#msg_block % 64 == 0)
   return msg_block
end

-- given a block (or blocks), create the appropriate table
function blocks_to_btable(blocks)
   local bnum = #blocks % 64
   local current_block = 0
   local start
   local blocks = {}

   while current_block < bnum do
      start = current_block * 64 + 1 -- damn offsets by 1
      current_block = current_block + 1 -- ruining my day

      local W = {}
      for t = 1, 16 do
         W[t] = bytes_to_word(blocks:byte(start, start+3))
         start = start + 4
      end
      for t = 17,80 do
         W[t] = 0 -- set these all as 0 for now
      end
      blocks[current_block] = W
   end
   return blocks
end

-- blocks is a table of tables
-- outer tables are tables
-- inner tables  blocks[i] start with block M(i) (words 0 to 15 in a block),
--   then have empty values for spaces 16-79 which we fill in
-- wow is that confusing
-- just ask if you have questions (832)-570-9327
function sha1(blocks)
   local H0 = 0x67452301
   local H1 = 0xEFCDAB89
   local H2 = 0x98BADCFE
   local H3 = 0x10325476
   local H4 = 0xC3D2E1F0
   for index,value in ipairs(blocks) do
         local w = value
         for t=16,79 do
            w[t+1] = rotate_word(bxorw[t+1-3],
                                  (bxor(w[t+1-8],
                                        (bxor(w[t+1-14],
                                              w[t+1-16])))), 1)
         end
         local A,B,C,D,E = H0,H1,H2,H3,H4
         for t=0,80 do
            local temp = rotate_word(A,5) + f(t+1,B,C,D) + E + w[t+1]+k(t+1)
            E,D,C,B,A = D,C,rotate_word(B,30),A,temp
         end
         H0,H1,H2,H3,H4 = H0+A, H1+B, H2+C, H3+D, H4
   end
   return {H0,H1,H2,H3,H4}
end

function file_exists(file)
   local f = io.open(file, "rb")
   if f then f:close() end
   return f ~= nil
end

function main()
   local lines = {}
   local file = arg[1]
   if not file_exists(file) then return {} end
   for line in io.lines(file) do
         table.insert(lines, line)
   end
   print(#lines, "messages total")
   local s = os.clock()

   for i,l in pairs(lines) do
         local msg = l
         local msgblocks = msg_to_blocks(msg)
         local msgbtable = blocks_to_btable(msgblocks)
         local msghashtable = sha1(msgbtable)
   end
   local e = os.clock()
   print(string.format("Hash time: %.2f\n", e-s))
end

main()
