import torchvision
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import random

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class RandomRotate():
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, mask):
        angle = random.randint(-self.angle, self.angle)

        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)

        return image, mask
    
class RandomHFlip():
    def __init__(self, p = 0.5):
        self.p = p
        
    def __call__(self, image, mask):
        sp = random.random()
        if sp < self.p:
            mask = TF.hflip(mask)
            image = TF.hflip(image)
        return image, mask
    
class RandomGaussianBlur():
    def __init__(self, p = 0.5, kernel_size = 9):
        self.p = p
        self.kernel_size = kernel_size
        
    def __call__(self, image, mask):
        sp = random.random()
        if sp < self.p:
            ks = random.randint(0, self.kernel_size)
            ks = ks+1 if ks % 2 == 0 else ks
            image = TF.gaussian_blur(image, ks)
            
        return image, mask
    
class RandomCrop():
    def __init__(self, min_scale = 0.9):
        self.min_scale = min_scale
        
    def __call__(self, image, mask):
        
        scale = random.uniform(self.min_scale, 1)
        
        h = int(scale * image.shape[1])
        w = int(scale * image.shape[2])
        
        y1 = random.randint(0, image.shape[1]-h)
        x1 = random.randint(0, image.shape[2]-w)
        
        image = image[:, y1:y1+h, x1:x1+h]
        mask = mask[:, y1:y1+h, x1:x1+h]
        
        return image, mask

class RandomCropToAspect():
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self, image, mask):
        h, w = self.shape
        ratio = h / w
        
        h_img, w_img = image.shape[1], image.shape[2]
        ratio_img = h_img / w_img
        
        if ratio_img > ratio:
            # height in image is bigger and should be reduced
            h_new = int(h * w_img / w)
            y = random.randint(0, h_img - h_new)
            image = image[:, y:y+h_new, :]
            mask = mask[:, y:y+h_new, :]
        else:
            w_new = int(w * h_img / h)
            x = random.randint(0, w_img - w_new)
            image = image[:, :, x:x+w_new]
            mask = mask[:, :, x:x+w_new]
            
        return image, mask
    
class Resize():
    def __init__(self, shape):
        self.shape = shape
        
    def __call__(self, image, mask):
        
        image = TF.resize(image, self.shape, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.shape, interpolation=InterpolationMode.NEAREST)
        
        return image, mask