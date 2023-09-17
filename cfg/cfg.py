img_norm_cfg = dict(
    mean=[255 * 0.485, 255 * 0.456, 255 * 0.406],
    std=[255 * 0.229, 255 * 0.224, 255 * 0.225],
    to_rgb=False,
)



train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize",size=(224,224)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type="ImageToTensor",keys=['img'])
]


trian_data = dict(
    cosal_paths=[r'C:/Users/Administrator/Desktop/CoSal2015'],
    batch_size=2,
    group_size=2,
    sal_batch_size=3,
    sal_paths=[r'C:/Users/Administrator/Desktop/CoSal2015'],
    shuffle=True,
    num_workers=2,
    pipeline=train_pipeline,
    
)


model = dict(
    backbone=dict(type='VGG16')
)
