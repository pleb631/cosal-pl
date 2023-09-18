import argparse
import os.path as osp
from mmcv import Config,DictAction

from dataset import build_dataloader
from models import BaseSEG
import lightning.pytorch as pl




def parse_args():
    parser = argparse.ArgumentParser(description="cosal")
    parser.add_argument("config", type=str, help="train config file path")
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
 
    return parser.parse_args()  

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    datasets = build_dataloader(**cfg.trian_data)
    model = BaseSEG(**cfg.model)
    # from pprint import pprint
    # for _ in range(1):
    #     for i in datasets:
    #         pprint(i['sal_img'].shape)
    #         break
    
    # ckpt_callback = pl.callbacks.ModelCheckpoint(
    # monitor='loss',
    # save_top_k=1,
    # mode='min'
    # )
    # early_stopping = pl.callbacks.EarlyStopping(monitor = 'val_loss',
    #            patience=3,
    #            mode = 'min')
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
     monitor='loss',
     dirpath='tb_logs/my_model/',
     filename='sample-mnist-{epoch:02d}-{loss:.2f}',
     every_n_epochs=1
 )
    logger=pl.loggers.TensorBoardLogger("tb_logs", name="my_model")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(max_epochs=10,callbacks=[lr_monitor,checkpoint_callback],logger=logger)
    trainer.fit(model=model, train_dataloaders=datasets)
    print("done!!")


    

if __name__ == "__main__":
    main()