local function try_loading(name, extension, prefix)
   local path = name
   prefix = prefix or ''
   if extension then path = './includes/' .. name .. '.' .. extension end
   local f = package.loadlib(path, prefix .. 'luaopen_' .. name)
   if f then f() end
   return f ~= nil
end

local function load_gl_lib(name)
   local ok = try_loading('luagl', 'so') or
              try_loading(name, 'bundle') or
              try_loading(name, 'bundle', '_') or
              try_loading(name, 'dll') or
              try_loading(name)
   
   if not ok then
      print('Cannot load ' .. name)
   end
end

load_gl_lib('luagl')
load_gl_lib('luaglut')
load_gl_lib('memarray')