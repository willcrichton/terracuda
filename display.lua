require "luagl"
renderer = require "renderer"

local window
local should_quit = false
local width = 400
local height = 400

function display()
   if should_quit then return end

   -- TODO: get the image from the renderer 

   glDisable(GL_DEPTH_TEST)
   glClearColor(0, 0, 0, 1)
   glClear(GL_COLOR_BUFFER_BIT)

   glMatrixMode(GL_PROJECTION)
   glLoadIdentity()
   glOrtho(0, width, 0, height, -1, 1)

   glMatrixMode(GL_MODELVIEW)
   glLoadIdentity()
   glRasterPos2d(0, 0)
   -- glDrawPixels(w, h, GL_RGBA, GL_FLOAT, data)

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

function main()
   print(renderer.foo)

   glutInit(arg)
   glutInitWindowSize(width, height)
   glutInitDisplayMode(GLUT_RGB + GLUT_DOUBLE + GLUT_DEPTH)
   window = glutCreateWindow("Terracuda")
   glutDisplayFunc(display)
   glutKeyboardFunc(keyboard)
   glutMainLoop()
end

if not pcall(debug.getlocal, 4, 1) then
   main()
end
