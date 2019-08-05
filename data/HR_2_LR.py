from PIL import Image
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp'])


def h2l(HRim,scale_factor):#下采样再上采样
    w,h=HRim.size
    LRim=HRim.resize((w//scale_factor,h//scale_factor),Image.BICUBIC)
    LRim=LRim.resize((w,h),Image.BICUBIC)
    return LRim


if __name__=="__main__":
    HR_dir = "../Set5"
    HRimage_filenames = [os.path.join(HR_dir, x) for x in os.listdir(HR_dir) if is_image_file(x)]
    HRimage_filenames.sort()
    scales = [2, 3, 4]
    for HRdir in HRimage_filenames:
        (filepath, tempfilename) = os.path.split(HRdir)
        (filename, extension) = os.path.splitext(tempfilename)
        HRim = Image.open(HRdir)
        for scale in scales:
            LRim=h2l(HRim,scale_factor=scale)
            LRname = os.path.join(filepath,filename+"_scale_"+str(scale)+extension)
            LRim.save(LRname)

