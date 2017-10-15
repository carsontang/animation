import cv2
import numpy as np
import subprocess as sp

from tqdm import tqdm

import utils

FFMPEG_BIN = "ffmpeg"

class Animation(object):

    def __init__(self, fps, duration, img1, img2, effect):
        self.fps = fps
        self.duration = duration
        self.effect = effect
        self.img1 = img1
        self.img2 = img2

    def animate(self, outfile):
        command = [
            FFMPEG_BIN,
            '-y',  # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '%dx%d' % (1280, 720),  # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),  # frames per second
            '-i', '-',  # The imput comes from a pipe
            '-an',  # Tells FFMPEG not to expect any audio
            '-vcodec', 'mpeg',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'error',
            outfile,
        ]

        writing_process = sp.Popen(command, stdin=sp.PIPE)

        for frame in tqdm(self.effect(self.img1, self.img2, duration=self.duration, fps=self.fps), desc="Animating"):
            writing_process.stdin.write(frame)

        writing_process.stdin.close()
        writing_process.wait()
        print('Done animating')


def curtain(from_img, to_img, duration=10, fps=30):
    num_frames = duration * fps
    height, width = from_img.shape

    for y in range(0, height, height // num_frames):
        new_img = np.zeros_like(from_img)
        new_img[0:y, :] = from_img[0:y, :]
        new_img[y:, :] = to_img[y:, :]
        yield cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    for y in range(2 * fps):
        yield cv2.cvtColor(np.zeros_like(from_img).astype(np.uint8), cv2.COLOR_GRAY2RGB)

def transition(from_img, to_img, duration=10, fps=30):
    num_frames = duration * fps # Number of frames needed to produce a video of length duration with fps
    for alpha in np.linspace(0.0, 1.0, num_frames):
        blended = utils.blend(from_img, to_img, alpha)
        yield cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_GRAY2RGB)

if __name__ == '__main__':

    img1 = cv2.cvtColor(cv2.imread('./frame_1112.jpg'), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread('./frame_1162.jpg'), cv2.COLOR_BGR2GRAY)

    ani = Animation(effect=curtain, img1=img1, img2=img2, fps=60, duration=2)
    ani.animate("vertical_curtain.mp4")

    ani2 = Animation(effect=transition, img1=img1, img2=img2, fps=60, duration=2)
    ani2.animate("transition.mp4")
