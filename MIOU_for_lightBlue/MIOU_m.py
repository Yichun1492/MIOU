import argparse
import os 
import sys 
import numpy as np
from os.path import join
from PIL import Image
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n) 
    #print(list(set(a[(a > 0) & (a < n)])))
    #print(np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).shape)
    #print("shape:",list(a).count(14))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

#def label_mapping()

def per_class_iu(hist):
    #print(np.diag(hist))
    #print(hist[0])
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)-hist[0])

def main():
    
    parser = argparse.ArgumentParser(description="Calculate miou for each classes")
    parser.add_argument("--pd_dir",dest="predict_dir",type=str,help="dir for predict results ")
    parser.add_argument("--gt_dir",dest="groundtruth_dir",type=str,help="dir for groundtruth ")
    parser.add_argument("--name_classes",dest="name_classes_str",type=str,help="str for name classes",default="0/1.bed_set/2.book/3.ceiling/4.chair_sofa/5.floor/6.furniture/7.door/8.picture/9.toilet/10.table/11.monitor/12.wall/13.window/14.person/15.others/16.stairs/17.light_source/18.plant/19.pillar/20.curtain/21.pillow")
    

    #parser.add_help("Hi")
    args=parser.parse_args()
    gt_dir_abspath=os.path.abspath(args.groundtruth_dir)
    pd_dir_abspath=os.path.abspath(args.predict_dir)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    num_classes=22
    name_classes=[]
    for i in args.name_classes_str.split("/"):
        name_classes.append(i)

    hist = np.zeros((len(name_classes),len(name_classes)))

    for path,dirs,files in os.walk(gt_dir_abspath):
        for file in files:
            label=np.array(Image.open(join(path,file))) #530x730=386900
            file= file.split(".")[0]+str(".448_336_0.000000.png")
            pred=np.array(Image.open(join(pd_dir_abspath,file))) #530x730=386900
            print(file,)
            if len(label.flatten()) != len(pred.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
    #print(hist[0][14])        
    mIoUs = per_class_iu(hist)
    
    print('val mIOU 17 classes:\t' + str(round(np.nansum(mIoUs)/17 * 100, 10))+"%")
    print('val mIOU:\t' + str(round(np.nansum(mIoUs)/21 * 100, 10))+"%")
    print("")
    for ind_class in range(num_classes):
        if ind_class ==0: continue
        print(name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 8))+"%")
    #print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))) 
    
if __name__ == '__main__':
    main()