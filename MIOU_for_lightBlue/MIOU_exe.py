import argparse
import os 
import sys 
import numpy as np
from os.path import join
from PIL import Image
import json

def fast_hist(a, b, n): #a:label b:predict
    k = (a >= 0) & (a <= n) 
    #print(list(set(b[(b >= 0) & (b <= n)])))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)-hist[0])

def pred_mapping(pd_mat,mapping):
    output = np.copy(pd_mat)
    for ind in range(len(mapping)):
        output[pd_mat == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def main():
    with open('.\\info.json', 'r') as fp: 
        info = json.load(fp) 
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)

    parser = argparse.ArgumentParser(description="Calculate miou for each classes")
    parser.add_argument("--pd_dir",dest="predict_dir",type=str,help="dir for predict results ")
    parser.add_argument("--gt_dir",dest="groundtruth_dir",type=str,help="dir for groundtruth ")
    parser.add_argument("--remap",dest="remap",type=str,help="0:Remap Pred ,1:Remap Label")
    parser.add_argument("--imgFormat",dest="imgFormat",type=str,help="Example: .448_336_0.000000.png")
    args=parser.parse_args()
    gt_dir_abspath=os.path.abspath(args.groundtruth_dir)
    pd_dir_abspath=os.path.abspath(args.predict_dir)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    hist = np.zeros((len(name_classes),len(name_classes)))
    for path,dirs,files in os.walk(gt_dir_abspath):
        for file in files:
            label=np.array(Image.open(join(path,file))) #530x730=386900
            file =file.split(".")[0]+args.imgFormat
            pred=np.array(Image.open(join(pd_dir_abspath,file))) #530x730=386900
            #print("pred before maping : ",list(set(pred[(pred > 0)])))
            if int(args.remap) == 0 :
                pred = pred_mapping(pred, mapping)
            else:
                label = pred_mapping(label, mapping)
            #print(file,)
            #if len(label.flatten()) != len(pred.flatten()):
            #    print('Skipping: len(gt) = {:d}, len(pred) = {:d}'.format(len(label.flatten()), len(pred.flatten())))
            #    continue
            #print(len(label.flatten()),len(pred.flatten()))
            hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
            #break
    #print(hist.shape)        
    #print(hist)
    mIoUs = per_class_iu(hist)
    
    print('val mIOU 17 classes:\t' + str(round(np.nansum(mIoUs)/2 * 100, 4))+"%")
    #print('val mIOU:\t' + str(round(np.nansum(mIoUs)/21 * 100, 10))+"%")
    print("")
    for ind_class in range(num_classes):
        if ind_class ==0: continue
        print(name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 4))+"%")
    #print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2))) 
    f = open("confusion.txt","w")
    for ind_class in range(num_classes+1):
        if ind_class ==0: 
            f.write("Classes")
            continue
        if ind_class ==1:
            #f.write("")
            #print("\t\t",end="")
            for ind in range(num_classes):
                if ind == 0: continue
                f.write("\t"+name_classes[ind])
                #print(str(ind)+'    ',end="")
        else:
            f.write("\n")
            f.write(name_classes[ind_class-1])
            #print(name_classes[ind_class-1],end="\t")
            for i in range(1,num_classes):
                f.write("\t"+str(hist[ind_class-1][i]))
                #print('(5.2%)' % float(hist[ind_class-1][i]))
        
    f.write("\n")
    f.close()
if __name__ == '__main__':
    main()