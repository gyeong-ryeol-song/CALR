import os
import traceback
from config import get_configs
from train_resnet import run_train_resnet
from train_q2l import run_train_q2l
import torch



def main():
    
    P, args = get_configs()
    print(P, '\n')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = P['gpu_num']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)   
    print('Count of using GPUs:', torch.cuda.device_count())   
    print('Current cuda device:', torch.cuda.current_device())


    if P['mode'] == 'train_resnet':
        print('###### Train ssl start ######')   
        run_train_resnet(P)
    elif P['mode'] == 'train_q2l':
        print('###### Train ssl start ######') 
        run_train_q2l(P,args)
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())

