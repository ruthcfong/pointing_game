import numpy as np
from skimage.transform import resize
from scipy.ndimage import zoom
import torch

from torchvision import transforms


def get_caffe_transform(size,
                        bgr_mean=[103.939, 116.779, 123.68],
                        scale=255):
    """Return a composition of transforms that replicates the caffe data
       transformation pipeline.
    """

    # Compose transform that replicates the order of transforms in caffe.io.
    transform = transforms.Compose([
        PILToNumpy(preserve_range=False, dtype=np.float32),
        CaffeResize(size),
        CaffeTranspose((2, 0, 1)),
        CaffeChannelSwap((2, 1, 0)),
        CaffeScale(scale),
        CaffeNormalize(bgr_mean),
        NumpyToTensor(),
    ])

    return transform


class PILToNumpy(object):
    """Converts PIL image to numpy array.
       Default behavior: change to numpy float32 array between [0,1].
    """
    def __init__(self, preserve_range=False, dtype=np.float32):
        self.preserve_range = preserve_range
        self.dtype = dtype

    def __call__(self, x):
        # assert isinstance(x, Image)
        x = np.array(x, dtype=self.dtype)
        if not self.preserve_range:
            x /= 255.
        return x


class NumpyToTensor(object):
    """Converts numpy array to PyTorch tensor."""
    def __call__(self, img):
        x = torch.from_numpy(img)
        return x


class CaffeResize(object):
    """Equivalent to caffe.io.resize_image if size = (height, width);
       expects a numpy array in (H, W, C) order.
    """
    def __init__(self, size, interp_order=1):
        assert(isinstance(size, tuple)
               or isinstance(size, list)
               or isinstance(size, int))
        self.size = size
        self.interp_order = interp_order

    def __call__(self, im):
        assert isinstance(im, np.ndarray)
        assert im.ndim == 3
        # Resize smaller side to size if size is an integer.
        if isinstance(self.size, int):
            h, w, _ = im.shape
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
            else:
                oh = self.size
                ow = int(self.size * w / h)
            size = (oh, ow)
        # Otherwise, resize image to height.
        else:
            assert len(self.size) == 2
            size = self.size
        if im.shape[-1] == 1 or im.shape[-1] == 3:
            im_min, im_max = im.min(), im.max()
            if im_max > im_min:
                # skimage is fast but only understands {1,3} channel images
                # in [0, 1].
                im_std = (im - im_min) / (im_max - im_min)
                resized_std = resize(im_std, size, order=self.interp_order,
                                     mode='constant')
                resized_im = resized_std * (im_max - im_min) + im_min
            else:
                # the image is a constant -- avoid divide by 0
                ret = np.empty((size[0], size[1], im.shape[-1]),
                               dtype=np.float32)
                ret.fill(im_min)
                return ret
        else:
            # ndimage interpolates anything but more slowly.
            scale = tuple(np.array(size, dtype=float) / np.array(im.shape[:2]))
            resized_im = zoom(im, scale + (1,), order=self.interp_order)
        return resized_im.astype(np.float32)


class CaffeTranspose(object):
    """Equivalent to caffe.io.set_transpose (default: (H,W,C) => (C,H,W))."""
    def __init__(self, order=(2, 0, 1)):
        self.order = order

    def __call__(self, x):
        if len(self.order) != x.ndim:
            raise Exception('Transpose order needs to have the same number of '
                            'dimensions as the input.')
        y = x.transpose(self.order)
        return x.transpose(self.order)


class CaffeChannelSwap(object):
    """Equivalent to caffe.io.set_channel_swap.
       Default behavior: RGB <=> BGR. Assumes (C,H,W) format.
    """
    def __init__(self, order=(2, 1, 0)):
        self.order = order

    def __call__(self, orig_img):
        assert(isinstance(orig_img, np.ndarray)
               or isinstance(orig_img, torch.Tensor))
        assert(len(orig_img.shape) == 3)
        if len(self.order) != orig_img.shape[0]:
            raise Exception('Channel swap needs to have the same number of '
                            'dimensions as the input channels.')
        new_img = orig_img[self.order, :, :]
        return new_img


class CaffeScale(object):
    """Equivalent to caffe.io.set_raw_scale."""
    def __init__(self, scale):
        assert isinstance(scale, int) or isinstance(scale, float)
        self.scale = scale

    def __call__(self, x):
        assert isinstance(x, np.ndarray)
        return x * self.scale


class CaffeNormalize(object):
    """Equivalent to caffe.io.set_mean for """
    def __init__(self, mean):
        if isinstance(mean, list):
            mean = np.array(mean)
        assert isinstance(mean, np.ndarray)
        if mean.ndim == 1:
            mean = mean[:, np.newaxis, np.newaxis]
        else:
            assert False
        self.mean = mean

    def __call__(self, x):
        if self.mean.shape[0] != x.shape[0]:
            raise ValueError('Mean channels incompatible with input.')
        x -= self.mean
        return x
