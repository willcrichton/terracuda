require "luagl"
local cuda = terralib.require("cudalib")

struct Params { 
   radius: &double,
   position: &double,
   color: &double,
   num_circles : int, 
   width : int,
   height : int,
   data : &double }

local window
local frame_time
local renderer
local width, height = 800, 800
local data = memarray('float', width * height * 4)
local params = terralib.global(Params)

terra get_pixel(i : int)
   var pixel = params.data[i]
   params.data[i] = 0.0
   return pixel
end

function display()

   -- get pixel data from the given renderer
   renderer.get_image(params:get())
   for i = 0, width * height * 4 - 1 do
      data[i] = get_pixel(i)
   end

   glDisable(GL_DEPTH_TEST)
   glClearColor(0, 0, 0, 1)
   glClear(GL_COLOR_BUFFER_BIT)

   glMatrixMode(GL_PROJECTION)
   glLoadIdentity()
   glOrtho(0, width, 0, height, -1, 1)

   -- put the renderer's pixel data on screen
   glMatrixMode(GL_MODELVIEW)
   glLoadIdentity()
   glRasterPos2d(0, 0)
   glDrawPixels(width, height, GL_RGBA, GL_FLOAT, data:ptr())

   -- track draw time
   local new_time = os.clock()
   print(string.format("frame time: %.2f", new_time - frame_time))
   frame_time = new_time

   glutSwapBuffers()
   glutPostRedisplay()
end

function keyboard(key)
   if key == 27 then
      cuda.free(params.radius)
      cuda.free(params.color)
      cuda.free(params.position)
      cuda.free(params.data)

      glutDestroyWindow(window)
      os.exit(0)
   end
end

local C = terralib.includec('stdlib.h')
terra update_params(w : int, h : int)
   params.width = w
   params.height = h
   params.data = [&double](cuda.alloc(sizeof(double) * 4 * w * h))
end

function resize(w, h)
   width = w
   height = h
   data = memarray('float', w * h * 4)

   update_params(width, height)

   glViewport(0, 0, w, h)
   glutPostRedisplay()
end

terra rand()
   return (C.rand() % 10000) / 10000.0
end

terra load_scene(scene : int)
   if scene == 0 then
      params.radius = [&double](cuda.alloc(sizeof(double) * 3))
      params.position = [&double](cuda.alloc(sizeof(double) * 9))
      params.color = [&double](cuda.alloc(sizeof(double) * 9))

      params.radius[0] = 0.3
      params.radius[1] = 0.3
      params.radius[2] = 0.3

      params.position[0] = 0.4
      params.position[1] = 0.5
      params.position[2] = 0.75
      params.position[3] = 0.5
      params.position[4] = 0.5
      params.position[5] = 0.5
      params.position[6] = 0.6
      params.position[7] = 0.5
      params.position[8] = 0.25

      params.color[0] = 1.0
      params.color[1] = 0.0
      params.color[2] = 0.0
      params.color[3] = 0.0
      params.color[4] = 1.0
      params.color[5] = 0.0
      params.color[6] = 0.0
      params.color[7] = 0.0
      params.color[8] = 1.0

      params.num_circles = 3

   elseif scene == 1 then
      var N = 1000
      var depths : double[1000]

      for i = 0, N do
         depths[i] = rand()
      end

      for i = 0, N do
         for j = i + 1, N do
            if depths[j] < depths[i] then
               depths[i], depths[j] = depths[j], depths[i]
            end
         end
      end

      params.radius = [&double](cuda.alloc(sizeof(double) * N))
      params.position = [&double](cuda.alloc(sizeof(double) * 3 * N))
      params.color = [&double](cuda.alloc(sizeof(double) * 3 * N))
      params.num_circles = N

      for i = 0, N do
         params.radius[i] = 0.02 + 0.06 * rand()

         params.position[3 * i] = rand()
         params.position[3 * i + 1] = rand()
         params.position[3 * i + 2] = depths[i]

         params.color[3 * i] = 0.1 + 0.9 * rand()
         params.color[3 * i + 1] = 0.2 + 0.5 * rand()
         params.color[3 * i + 2] = 0.5 + 0.5 * rand()
      end
   end
end

function main()
   glutInit(arg)
   glutInitWindowSize(width, height)
   glutInitDisplayMode(GLUT_RGBA + GLUT_DOUBLE)
   window = glutCreateWindow("Terracuda")

   glutDisplayFunc(display)
   glutKeyboardFunc(keyboard)
   glutReshapeFunc(resize)

   load_scene(tonumber(arg[1]))
   frame_time = os.clock()

   if not arg[2] or arg[2] == "cuda" then
      renderer = terralib.require("cuda_renderer")
   else
      renderer = terralib.require("serial_renderer")
   end

   glutMainLoop()
end

if not pcall(debug.getlocal, 4, 1) then
   main()
end
