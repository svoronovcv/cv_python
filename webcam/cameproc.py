import cv2
from managers import WindowManager, CaptureManager

class Camproc(object):

    def __init__(self):
        self._windowManager = WindowManager('Camproc', self.onKeyPress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0),
                                              self._windowManager, True)

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        if keycode == 32: #space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: #tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: #escape
            self._windowmanager.destroyWindow()

if __name__ == '__main__':
    Camproc().run()
