from __future__ import absolute_import, division, print_function

from skimage import io, color, transform, img_as_float
import os, glob, urllib, hashlib, time, pickle
import numpy as np


def load_data(name, sizes=(100, 10, 10), is_random=False):
    if not is_random:
        np.random.seed(23707) # ensures reproducibiliy
    else:
        np.random.seed(None)
    # load 90 random images
    image_paths = glob.glob("cropped/" + name + '[0-9]*.png')
    selected_paths = np.random.choice(image_paths, sum(sizes), replace=False)
    images = io.imread_collection(selected_paths).concatenate()
    # change to vector and ensure values are in range 0 to 1
    return img_as_float(images.reshape((sum(sizes), -1)))


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
    if not os.path.exists("uncropped/"):
        os.makedirs("uncropped/")
    if not os.path.exists("cropped/"):
        os.makedirs("cropped/")

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
                timeout(urllib.urlretrieve,
                        (line.split()[4], "uncropped/" + filename), {}, 30)
                # ensure that the image was successfully downloaded
                if not os.path.isfile("uncropped/" + filename):
                    continue

                # get the hash of the image and ensure it matches expected
                f = open("uncropped/" + filename, 'rb')
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
    os.chdir('uncropped/')
    all_images = set(glob.glob('*'))
    for file in all_images - valid_images:
        os.remove(file)
    os.chdir(cwd)

    # crop and resize images
    for filename, coord in cropping_guide.iteritems():
        # load the image as a greyscale image
        try:
            image = io.imread("uncropped/" + filename, as_grey=True)
            # crop and resize
            face = transform.resize(
                image[coord[1]:coord[3], coord[0]:coord[2]], (32, 32))
        except:
            continue
        # remove extension
        filename = filename[:filename.rfind('.')]
        io.imsave("cropped/" + filename + '.png', face)