img_norm_cfg = dict(
    mean=[255 * 0.485, 255 * 0.456, 255 * 0.406],
    std=[255 * 0.229, 255 * 0.224, 255 * 0.225],
    to_rgb=False,
)


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(224, 224)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
]


trian_data = dict(
    cosal_paths=[r"D:/Dataset/Dataset/COCO9213"],
    batch_size=2,
    group_size=5,
    sal_batch_size=8,
    sal_paths=[r"D:/Dataset/Dataset/COCOSAL"],
    shuffle=True,
    num_workers=2,
    pipeline=train_pipeline,
)
val_data = dict(
    cosal_paths=[r"D:/Dataset/Dataset/COCO9213"],
    batch_size=1,
    group_size=5,
    sal_batch_size=0,
    sal_paths=None,
    shuffle=False,
    num_workers=2,
    pipeline=train_pipeline,
)
train_set = dict(
    weight_decay=1e-4,
    lr=1e-4,
    lr_scheduler="multistep",
    gamma=0.1,
    milestones=[50,60],
)
model = dict(
    backbone=dict(type="VGG16", pretrained=True),
    aux_head=dict(type="sal_Decoder"),
    neck=dict(type="neck"),
    head = dict(type="cosal_decoder"),
    train_set=train_set,
)
