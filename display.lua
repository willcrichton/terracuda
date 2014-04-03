require "luagl"
renderer = require "renderer"

local window
local should_quit = false
local params = {
   width = 400,
   height = 400,
   circles = {}
}

function display()
   if should_quit then return end

   -- get pixel data from the given renderer
   renderer.get_image(params)

   glDisable(GL_DEPTH_TEST)
   glClearColor(0, 0, 0, 1)
   glClear(GL_COLOR_BUFFER_BIT)

   glMatrixMode(GL_PROJECTION)
   glLoadIdentity()
   glOrtho(0, params.width, 0, params.height, -1, 1)

   -- put the renderer's pixel data on screen
   glMatrixMode(GL_MODELVIEW)
   glLoadIdentity()
   glRasterPos2d(0, 0)
   glDrawPixels(params.width, params.height, GL_RGBA, GL_FLOAT, params.data:ptr())

   for i = 0, 4 * params.width * params.height - 1 do
      params.data[i] = 0
   end

   glutSwapBuffers()
   glutPostRedisplay()
end

function keyboard(key)
   if key == 27 then
      should_quit = true
      glutDestroyWindow(window)
      os.exit(0)
   end
end

function resize(w, h)
   params.width = w
   params.height = h
   params.data = memarray('float', w * h * 4)

   glViewport(0, 0, w, h)
   glutPostRedisplay()
end

function load_scene()
   params.circles = {
      {radius = 0.3,
       position = {0.4, 0.5, 0.75},
       color = {1.0, 0.0, 0.0}
      },
      {radius = 0.3,
       position = {0.5, 0.5, 0.5},
       color = {0.0, 1.0, 0.0}
      },
      {radius = 0.3,
       position = {0.6, 0.5, 0.25},
       color = {0.0, 0.0, 1.0}
      }
   }
end

function main()
   glutInit(arg)
   glutInitWindowSize(params.width, params.height)
   glutInitDisplayMode(GLUT_RGBA + GLUT_DOUBLE)
   window = glutCreateWindow("Terracuda")

   glutDisplayFunc(display)
   glutKeyboardFunc(keyboard)
   glutReshapeFunc(resize)

   load_scene()

   glutMainLoop()
end

if not pcall(debug.getlocal, 4, 1) then
   main()
end
