import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import torchvision


def intersect(box_a, box_b):
    if box_a is None or box_b is None:
        return 0
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    if box_a is None or box_b is None:
        return 0
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,labels=None):
        for t in self.transforms:
            img= t(img)
        # exit()
        return img


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class ConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32)


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if not boxes is None:
            height, width, channels = image.shape
            boxes[:, 0] *= width
            boxes[:, 2] *= width
            boxes[:, 1] *= height
            boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        if not boxes is None:
            height, width, channels = image.shape
            boxes[:, 0] /= width
            boxes[:, 2] /= width
            boxes[:, 1] /= height
            boxes[:, 3] /= height
        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image):
        # print("test1:",image.shape)
        return cv2.resize(image,(self.size,self.size))


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            # print("test:",swap)
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image


class ToCV2Image(object):
    def __call__(self, tensor):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0))


class ToTensor(object):
    def __call__(self, cvimage):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1)

class RandomCrop(object):
    def __init__(self,size=224):
        self.size=size
    def __call__(self,images):
        width,height,_=images.shape
        sw=random.randint(0,width-self.size)
        sh=random.randint(0,height-self.size)
        images=images[sw:sw+self.size,sh:sh+self.size]
        return images

class RandomMirror(object):
    def __call__(self, image):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[::-1, :]

        return image


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image):
        im = image.copy()
        im = self.rand_brightness(im)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im = distort(im)
        im=self.rand_light_noise(im)
        return im

class SSDAugmentation(object):
    def __init__(self, size=54, mean=(90, 90, 90)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            # ToAbsoluteCoords(),
            PhotometricDistort(),
            Resize(224),
            RandomCrop(self.size),
            RandomMirror(),
            ToTensor(),
            # ToPercentCoords(),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img):
        img=self.augment(img)
        return img

class Basetransform(object):
    def __init__(self, size=54, mean=(90, 90, 90)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ToTensor(),
            Resize((self.size,self.size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ToPercentCoords(),
            # SubtractMeans(self.mean)
        ])

    def __call__(self, img):
        img=self.augment(img)
        return img
#
# img=cv2.imread("example.jpg")
# transform=SSDAugmentation()
# img=transform(img)
# print(img)