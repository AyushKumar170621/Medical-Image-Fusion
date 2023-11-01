from flask import Flask, render_template, request
import glob
import os
import cv2
import time
import torch
import cv2,threading
import pywt
import pywt.data
from model import myIFCNN
from matplotlib.image import imread
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from utils.myTransforms import denorm, norms, detransformcv2
import numpy as np
app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('upload.html')
	
def applyDWT():
    images = glob.glob('static/*.tif')
    for x in images:
        original = imread(x)
        filename = x[7:11]
        im = Image.open(x)
        im.save('static/'+filename+'.jpeg')
        cA,cD = pywt.dwt(original, 'haar')
        cv2.imwrite('static/1drs/cA'+filename+'.png',cA)
        cv2.waitKey()
        cv2.imwrite('static/1drs/cD'+filename+'.png',cD)
        cv2.waitKey()

def mergedwtimage(model):
    dwtimage = glob.glob('static/1drs/*.png')
    for ind in range(0,len(dwtimage),2):
        filename = 'img'+str(ind//2)
        print(filename)
        path1 = os.path.join(dwtimage[ind])
        path2 = os.path.join(dwtimage[ind+1])
        is_save = True
        is_gray = True
        mean=[0, 0, 0]         # normalization parameters
        std=[1, 1, 1]
        from utils.myDatasets import ImagePair
        pair_loader = ImagePair(impath1=path1, impath2=path2, 
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean, std=std)
                                          ]))
        img1, img2 = pair_loader.get_pair()
        img1.unsqueeze_(0)
        img2.unsqueeze_(0)

        # perform image fusion
        with torch.no_grad():
            res = model(Variable(img1.cuda()), Variable(img2.cuda()))
            res = denorm(mean, std, res[0]).clamp(0, 1) * 255
            res_img = res.cpu().data.numpy().astype('uint8')
            img = res_img.transpose([1,2,0])

        # save fused images
        if is_save:
            if is_gray:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = Image.fromarray(img)
                img.save('static/1dmer/'+filename+'.png', format='PNG', compress_level=0)
            else:
                img = Image.fromarray(img)
                img.save('static/1dmer/'+filename+'.png', format='PNG', compress_level=0)


def applyIDWT():
    dwfin = glob.glob('static/1dmer/*.png')
    nca=cv2.imread(dwfin[0],cv2.IMREAD_GRAYSCALE)
    ncd=cv2.imread(dwfin[1],cv2.IMREAD_GRAYSCALE)
    res=pywt.idwt(nca,ncd, 'haar','symmetric')
    cv2.imwrite('static\\result\\res.png', res)
    cv2.waitKey()


@app.route('/extra',methods = ['GET','POST'])
def operation():
    if request.method == 'POST':
      f = request.files['file1']
      f2 = request.files['file2']
      f.save('static\\'+'img1.tif')
      f2.save('static\\'+'img2.tif')
      fscheme=int(request.form.get("fshm"))
    else:
        return "error"

    applyDWT()
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    fuse_scheme = fscheme
    if fuse_scheme == 0:
        model_name = 'IFCNN-MAX'
    elif fuse_scheme == 1:
        model_name = 'IFCNN-SUM'
    elif fuse_scheme == 2:
        model_name = 'IFCNN-MEAN'
    else:
        model_name = 'IFCNN-MAX'

    #load the model
    model = myIFCNN(fuse_scheme=fuse_scheme)
    model.load_state_dict(torch.load('snapshots/'+ model_name + '.pth'))
    model.eval()
    model = model.cuda()
    print(model)
    mergedwtimage(model)
    applyIDWT()
    return render_template('result.html')
if __name__ == '__main__':
   app.run(debug = True)