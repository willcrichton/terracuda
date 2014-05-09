struct Params {
   num_msgs: int,
   msgs: &uint32,
   results: &uint32}

local params = terralib.global(Params)
local sha = terralib.require("sha1")

local lines = {}

local cuda = terralib.require("cudalib")
local C = terralib.includec('stdlib.h')
local Cio = terralib.includec('stdio.h')

terra bytes_to_word(a : uint8, b : uint8, c : uint8, d : uint8)
   return a*0x1000000 + b*0x10000 + c*0x100 + d
end

terra hash_all(n : int)
   sha.hash(params)
end

terra pack_and_send(msg : &int8, idx : int, len : int, total : int)
   --Cio.printf("Original message: %s\n", msg)
   var tmp_msg : uint32[16]
   var msg_idx = 16*(idx-1)

   var zeroed_msg : uint8[64]
   for i=0,63 do
      if i < len then
         zeroed_msg[i] = msg[i]
      else
         zeroed_msg[i] = 0
      end
   end
   zeroed_msg[len] = 0x80

   for i=0,16 do
      var j = [uint32](bytes_to_word(zeroed_msg[4*i],
                                     zeroed_msg[4*i+1],
                                     zeroed_msg[4*i+2],
                                     zeroed_msg[4*i+3]))
      tmp_msg[i] = j
   end
   tmp_msg[15] = len*8
   for i=0,16 do
      --[[
      if i%4 ==0 and i > 0 then
         Cio.printf("\n")
      end
      Cio.printf("0x%x\t", tmp_msg[i])
      --]]
      params.msgs[msg_idx+i] = tmp_msg[i]
   end
end

terra setup_params(num_msgs : int)
   Cio.printf("%d messages total.\n", num_msgs)
   params.num_msgs = num_msgs
   params.msgs = [&uint32](cuda.alloc(sizeof(uint32)*16*num_msgs))
   -- every message has a 5 word (32 bit) message digest
   params.results = [&uint32](cuda.alloc(sizeof(int32)*5*num_msgs))
end

function prepack_input()
   lines = {}
   local file = arg[1]
   local total_msgs = 0
   if not file_exists(file) then return {} end
   for line in io.lines(file) do
         table.insert(lines, line)
         total_msgs = total_msgs + 1
   end
   setup_params(total_msgs)
   for i,line in ipairs(lines) do
         --print("\n", i, "\n")
         pack_and_send(line, i, string.len(line), total_msgs)
   end
end

terra print_results(total : int)
   Cio.printf("\nResults:\n\n")
   for i=0,5*total do
      if i%5 == 0 and i > 0 then
         Cio.printf("\n")
      end
      Cio.printf("%x\t", params.results[i])
   end
   Cio.printf("\n")
end

function file_exists(file)
   local f = io.open(file, "rb")
   if f then f:close() end
   return f ~= nil
end

function main()
   prepack_input()
   hash_all(#lines)
   print("Done")
   print_results(#lines)
   cuda.free(params.msgs)
   cuda.free(params.results)
end

main()

