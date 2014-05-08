struct Params {
   num_blocks: int,
   num_msgs: int,
   block_lengths: &int,
   start_block: &int,
   words: &int,
   result: &int}

local char,rep = string.char, string.rep
local modf = math.modf

local params = terralib.global(Params)
local sha = terralib.require("sha1")
local total_msgs = 0
local total_blocks = 0
local total_len = 0

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
terra pack_and_send(lines : &int8)
   params.num_msgs = [#lines]
   params.num_blocks = total_blocks
   params.num_msgs = total_msgs
   params.block_lengths = [&int](cuda.alloc(sizeof(int)*total_msgs))
   params.start_block = [&int](cuda.alloc(sizeof(int)*total_msgs))
   params.words = [&int](cuda.alloc(sizeof(int)*total_len))
   -- every message has a 5 word result
   params.results = [&int](cuda.alloc(sizeof(int)*total_msgs*5))
   var cur_msg = 0
   var cur_block = 0
   var word_index = 0

   for i,v in pairs(lines) do
         -- blocks contains the padded message
         params.start_block[cur_msg] = cur_block
         var blocks = msg_to_blocks(v)
         for i=0,([#blocks] / 4) do
            if (word_index)%16 == 0 then
               cur_block = cur_block + 1
            end
            -- most important part
            var word = bytes_to_word(blocks:byte(word_index, word_index+3))
            print(word, word_index)
            params.words[word_index] = word
            word_index = word_index + 1
         end
         params.block_lengths[cur_msg] = [#blocks] / 16
         cur_block = cur_block + 1
         cur_msg = cur_msg + 1
   end
end

terra hash_all(n : int)
   sha.hash(params:get())

   -- print it out just to be sure it all hashed I guess
   for i=0,n do
         var res : &int = params.results[i*5]
         Cio.printf("Message #%d hash: (0x)%x %x %x %x %x\n",
                    i, res[0],res[1],res[2],res[3],res[4])
   end
end

function file_exists(file)
   local f = io.open(file, "rb")
   if f then f:close() end
   return f ~= nil
end

function main()
   local file = arg[1]
   if not file_exists(file) then return {} end
   local lines = {}
   for line in io.lines(file) do
         lines[#lines+1] = line
   end
   print("starting?")
   pack_and_send(lines)
   hash_all(#lines)
   print("Done?")
end

