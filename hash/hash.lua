struct Params {
   num_msgs: int,
   msgs: &int8,
   result: &int}

local char,rep = string.char, string.rep
local modf = math.modf

local params = terralib.global(Params)
local sha = terralib.require("sha1")
local total_msgs = 0
local total_blocks = 0
local total_len = 0

local lines = {}

local C = terralib.includec('stdlib.h')
local Cio = terralib.includec('stdio.h')

function bytes_to_word(a, b, c, d)
   return a*0x1000000 + b*0x10000 + c*0x100 + d
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
   total_blocks = total_blocks + (#msg_block / 64)
   total_len = total_len + (#msg_blocks / 4)
   total_msgs = total_msgs + 1
   return msg_block
end

-- take all messages and pad/pack them into the params struct
terra pack_and_send()
   -- Pad all of the messages out to 512 bits (64 bytes) with 0s
   Cio.printf("%s\n", )
   -- TODO: put all of the messages in the params struct
end

terra hash_all(n : int)
   sha.hash(params)
end

function file_exists(file)
   local f = io.open(file, "rb")
   if f then f:close() end
   return f ~= nil
end

function main()
   print("Hey there squirelly bear")
   params.num_msgs = 0
   local file = arg[1]
   if not file_exists(file) then return {} end
   for line in io.lines(file) do
         lines[#lines+1] = line
         params.num_msgs = params.num_msgs + 1
   end
   print(params.num_msgs)
   print("starting?")
   pack_and_send()
   hash_all(#lines)
   print("Done?")
end

main()

