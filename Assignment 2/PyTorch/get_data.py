import glob
import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from skimage import io, transform, img_as_float


def load_data(name, size, folder='bw32'):
    np.random.seed(10007)
    # load 90 random images
    image_paths = glob.glob(folder + '/' + name + '[0-9]*.png')
    selected_paths = np.random.choice(image_paths, size, replace=False)
    images = io.imread_collection(selected_paths).concatenate()
    # change to vector and ensure values are in range 0 to 1
    return img_as_float(images.reshape((size, -1)))


def timeout(func, args=(), kwargs=None, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    if kwargs is None:
        kwargs = {}
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result


def get_data():
    # Create a directory to store if one does not already exist
    if not os.path.exists("raw/"):
        os.makedirs("raw/")
    if not os.path.exists("bw32/"):
        os.makedirs("bw32/")
    if not os.path.exists("alexnet/"):
        os.makedirs("alexnet/")
    if not os.path.exists("bw64/"):
        os.makedirs("bw64/")

    # Since we need to open the file to check the hash and immediately opening a
    # file after closing is not guarenteed to work, deletion and cropping is
    # done at the end in batch using these two variables
    valid_images = set()
    cropping_guide = {}

    act = list(set([a.split("\t")[0] for a in
                    open("actor_image_data.txt").readlines()]))
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open("actor_image_data.txt"):
            if a in line:
                filename = name + str(i) + '.' + line.split()[4].split('.')[-1]

                # download the image
                timeout(urllib.request.urlretrieve,
                        (line.split()[4], "raw/" + filename), {}, 30)
                # ensure that the image was successfully downloaded
                if not os.path.isfile("raw/" + filename):
                    continue

                # get the hash of the image and ensure it matches expected
                f = open("raw/" + filename, 'rb')
                h = hashlib.sha256(f.read()).hexdigest()
                f.close()
                if h != line.split()[-1]:
                    continue

                print(filename)
                # store necessary information for later processing
                valid_images.add(filename)
                # [x1, y1, x2, y2]
                cropping_guide[filename] = list(
                    map(int, line.split()[5].split(',')))
                i += 1

    # remove all invalid images
    cwd = os.getcwd()
    os.chdir('raw/')
    all_images = set(glob.glob('*'))
    for file in all_images - valid_images:
        os.remove(file)
    os.chdir(cwd)

    # crop and resize images
    for filename, coord in cropping_guide.items():
        # load the image as a greyscale image
        try:
            image_bw = io.imread("raw/" + filename, as_grey=True)
            image_col = io.imread("raw/" + filename)
            # crop and resize
            face_bw32 = transform.resize(
                image_bw[coord[1]:coord[3], coord[0]:coord[2]], (32, 32))
            face_bw64 = transform.resize(
                image_bw[coord[1]:coord[3], coord[0]:coord[2]], (64, 64))
            face_alexnet = transform.resize(
                image_col[coord[1]:coord[3], coord[0]:coord[2]], (224, 224))
        except:
            continue
        # remove extension
        filename = filename[:filename.rfind('.')]
        io.imsave("bw32/" + filename + '.png', face_bw32)
        io.imsave("bw64/" + filename + '.png', face_bw64)
        io.imsave("alexnet/" + filename + '.png', face_alexnet)