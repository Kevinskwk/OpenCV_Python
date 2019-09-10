import cv2 as cv
import numpy as np
import os
from os import listdir 
from os.path import isfile, join
import glob


class Data_augmentation:
    def __init__(self, img_path, txt_path, image_name, txt_name):
        '''
        Import image
        :param img_path: Path to the image
        :param txt_path: Path to the label
        :param image_name: image name
        :param txt_name: label name
        '''
        self.img_path = img_path
        self.txt_path = txt_path
        self.name = image_name
        print(img_path + image_name)
        self.image = cv.imread(img_path+image_name)
        with open(txt_path+txt_name, 'r+') as f:
            read_data = f.read()
        self.txt = read_data

    def translate(self, image, txt, x_shf, y_shf):
        '''
        Translate image
        :param x_shf: x direction shift in pixels
        :param y_shf: y direction shift in pixels
        '''
        data_lines = txt.split('\n')
        data_lines.pop()
        data = [lines.split() for lines in data_lines]
        txt_out = ''
        w = image.shape[1]
        h = image.shape[0]
        #translate matrix
        M = np.array([[1,0,x_shf],[0,1,y_shf]],dtype=np.float32)
        #translate
        image = cv.warpAffine(image,M,(w,h))
        
        for line in data:
            x = float(line[1])+float(x_shf)/w
            y = float(line[2])+float(y_shf)/h
            string = '%s %s %s %s %s\n' % (line[0], x, y, line[3], line[4])
            txt_out += string 
        return image, txt_out
    
    def rotate(self, image, txt, angle=90, scale=1.0):
        '''
        Rotate the image
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        AUTO BOUNDING BOX SUPPORTS +-90 DEGREES ONLY!!!
        '''
        data_lines = txt.split('\n')
        data_lines.pop()
        data = [lines.split() for lines in data_lines]
        txt_out = ''
        w = image.shape[1]
        h = image.shape[0]
        #rotate matrix
        M = cv.getRotationMatrix2D((w/2,h/2), angle, scale)
        #rotate
        image = cv.warpAffine(image,M,(w,h))

        for line in data:
            y = 0.5-np.sin(angle/180.0*np.pi)*(float(line[1])-0.5)*w/h
            x = 0.5+np.sin(angle/180.0*np.pi)*(float(line[2])-0.5)*h/w
            width = scale*float(line[4])
            height = scale*float(line[3])
            string = '%s %s %s %s %s\n' % (line[0], x, y, width, height)
            txt_out += string
        return image, txt_out

    def flip(self, image, txt, vflip=False, hflip=False):
        '''
        Flip the image
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        data_lines = txt.split('\n')
        data_lines.pop()
        data = [lines.split() for lines in data_lines]
        txt_out = ''
        
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            elif vflip:
                c = 0
            else:
                c = 1    
            image = cv.flip(image, flipCode=c)

        for line in data:
            x = float(line[1])
            y = float(line[2])
            if hflip or vflip:
                if hflip and vflip:
                    x, y = 1.0-x, 1.0-y
                elif vflip:
                    y = 1.0-y
                else:
                    x = 1.0-x  
            string = '%s %s %s %s %s\n' % (line[0], x, y, line[3], line[4])
            txt_out += string
        return image, txt_out
    
    def resize(self, image, txt, x=1.5, y=1.5):
        '''
        Resize the image
        :param fx: x dimensional scaling factor
        :param fy: y dimensional scaling factor
        '''
        data_lines = txt.split('\n')
        data_lines.pop()
        data = [lines.split() for lines in data_lines]
        txt_out = ''
        image = cv.resize(image, None, fx=x, fy=y, interpolation=cv.INTER_CUBIC)
        for line in data:
            string = '%s %s %s %s %s\n' % (line[0], line[1], line[2], float(line[3])*x, float(line[4])*y)
            txt_out += string
        return image, txt_out 
    
    # Working in progress
    '''
    def general_affine_trans(self, image, txt, p_out=[[50,100],[200,50],[100,200]]):
        
        # General_affine_transformation
        # :param p_in: input points
        # :param p_out: y output points
        
        data = txt.split()
        w = image.shape[1]
        h = image.shape[0]
        pts_in = np.float32([[data[1]*w, data[2]*h], [(data[1]+data[3]/2)*w, data[2]*h], [data[1]*w, (data[2]+data[4]/2)*h]])
        pts_out = np.float32(p_out)

        M = cv.getAffineTransform(pts_in, pts_out)
        image = cv.warpAffine(image, M, (h, w))

        x, y = p_out[0][0]/w, p_out[0][1]/h
        width = (((p_out[1][0]-p_out[0][0])**2+(p_out[1][1]-p_out[0][1])**2)**0.5)*2/w
        height = (((p_out[2][0]-p_out[0][0])**2+(p_out[2][1]-p_out[0][1])**2)**0.5)*2/h
        string = '%s %s %s %s %s' % (data[0], x, y, width, height)
        return image, string
    '''
    # Working in progress
    '''
    def perspective_trans(self, image, p_in=[[56,65],[368,52],[28,387],[389,390]], p_out=[[0,0],[300,0],[0,300],[300,300]]):
        
        # Perspective_transformation
        # :param p_in: input points
        # :param p_out: y output points
        
        w = image.shape[1]
        h = image.shape[0]
        pts_in = np.float32(p_in)
        pts_out = np.float32(p_out)

        M = cv.getPerspectiveTransform(pts_in, pts_out)
        transformed = cv.warpPerspective(image, M, (h, w))
        return transformed
    '''

    def contrast_brightness(self, image, txt, alpha=1.5, beta=50):
        '''
        Change contrast and brightness
        :param alpha: contrast coefficient
        :param beta: brightness coefficient
        '''
        w = image.shape[1]
        h = image.shape[0]
        array_beta = cv.multiply(np.ones((h,w,3), dtype=np.float32),beta)
        # multiply every pixel value by alpha
        image = image*alpha
        # add a beta value to every pixel 
        image = image + array_beta
        return image, txt

    def gaussian_Blur(self, image, txt, kernal_x=7, kernal_y=7):
        '''
        Gaussian Blur
        :param kernal_x: x dimensional kernal_size MUST be odd number
        :param kernal_y: y dimensional kernal_size MUST be odd number
        '''
        blur = cv.GaussianBlur(image,(kernal_x,kernal_y),0)
        return blur, txt

    def add_GaussianNoise(self, image, txt, sigma=25):
        '''
        Gaussian Noise
        :param sigma: sigma coefficient for Gaussian Noise
        '''
        temp_image = np.float32(np.copy(image))
        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise = np.random.randn(h, w) * sigma

        noisy_image = np.zeros(temp_image.shape, np.float32)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:,:,0] = temp_image[:,:,0] + noise
            noisy_image[:,:,1] = temp_image[:,:,1] + noise
            noisy_image[:,:,2] = temp_image[:,:,2] + noise

        return noisy_image, txt

    def image_augment(self, save_path): 
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        name_int = self.name[:len(self.name)-4]
        # original
        img, txt = self.image.copy(), self.txt
        cv.imwrite(save_path+'%s' %str(name_int)+'_original.jpg', img)
        with open(save_path+'%s' %str(name_int)+'_original.txt', 'w+') as f:
            f.write(txt)

        # translation
        img_translate, txt_translate = self.translate(img, txt, 100, 10)
        cv.imwrite(save_path+'%s' %str(name_int)+'_translate.jpg', img_translate)
        with open(save_path+'%s' %str(name_int)+'_translate.txt', 'w+') as f:
            f.write(txt_translate)

        # rotation
        img_rot, txt_rot = self.rotate(img, txt)
        cv.imwrite(save_path+'%s' %str(name_int)+'_rot.jpg', img_rot)
        with open(save_path+'%s' %str(name_int)+'_rot.txt', 'w+') as f:
            f.write(txt_rot)

        # flipping
        img_flip, txt_flip = self.flip(img, txt, vflip=True, hflip=False)
        cv.imwrite(save_path+'%s' %str(name_int)+'_flip.jpg', img_flip)
        with open(save_path+'%s' %str(name_int)+'_flip.txt', 'w+') as f:
            f.write(txt_flip)

        # resizing
        img_resize, txt_resize = self.resize(img, txt)
        cv.imwrite(save_path+'%s' %str(name_int)+'_resize.jpg', img_resize)
        with open(save_path+'%s' %str(name_int)+'_resize.txt', 'w+') as f:
            f.write(txt_resize)

        # contrast and brightness
        img_cnb, txt_cnb = self.contrast_brightness(img, txt)
        cv.imwrite(save_path+'%s' %str(name_int)+'-ContrastBrightness.jpg', img_cnb)
        with open(save_path+'%s' %str(name_int)+'_ContrastBrightness.txt', 'w+') as f:
            f.write(txt_cnb)  

        # Gaussian blur
        img_blur, txt_blur = self.gaussian_Blur(img, txt)
        cv.imwrite(save_path+'%s' %str(name_int)+'_blur.jpg', img_blur)
        with open(save_path+'%s' %str(name_int)+'_blur.txt', 'w+') as f:
            f.write(txt_blur)

        # Gaussian noise
        img_noise, txt_noise = self.add_GaussianNoise(img, txt)
        cv.imwrite(save_path+'%s' %str(name_int)+'_GaussianNoise.jpg', img_noise)
        with open(save_path+'%s' %str(name_int)+'_GaussianNoise.txt', 'w+') as f:
            f.write(txt_noise)
    
# main
        
img_PATH = 'image/'
txt_PATH = 'label/'
save_PATH = 'image_aug/'

img_names = [f[len(img_PATH):] for f in glob.glob(img_PATH + '*.jpg')]
for i in range(len(img_names)):
    txt_names = img_names[i][:len(img_names[i])-4]+'.txt'
    image_i = Data_augmentation(img_PATH, txt_PATH, img_names[i], txt_names)
    image_i.image_augment(save_PATH)
