import argparse
import os.path as osp
from mmcv import Config

from dataset import build_dataloader
from models import BaseSEG
import lightning.pytorch as pl



def parse_args():
    parser = argparse.ArgumentParser(description="detect_val_tool")
    parser.add_argument("config", type=str, help="train config file path")
 
    return parser.parse_args()  

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    datasets = build_dataloader(**cfg.trian_data)
    model = BaseSEG(**cfg.model)
    # from pprint import pprint
    # for _ in range(1):
    #     for i in datasets:
    #         pprint(i['sal_img'].shape)
    #         break
        
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model=model, train_dataloaders=datasets)


    

if __name__ == "__main__":
    main()