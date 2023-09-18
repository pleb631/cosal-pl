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
    cosal_paths=[r"/root/autodl-tmp/Dataset/COCO9213"],
    batch_size=1,
    group_size=2,
    sal_batch_size=30,
    sal_paths=[r"/root/autodl-tmp/Dataset/COCOSAL"],
    shuffle=True,
    num_workers=8,
    pipeline=train_pipeline,
)
test_data = dict(
    cosal_paths=[r"C:\Users\Administrator\Desktop\CoSal2015"],
    batch_size=1,
    group_size=2,
    sal_batch_size=1,
    sal_paths=[r"C:\Users\Administrator\Desktop\CoSal2015"],
    shuffle=True,
    num_workers=2,
    pipeline=train_pipeline,
)
train_set = dict(weight_decay=1e-6,lr=1e-4,lr_scheduler="cosine",T_max=10,decay_rate=0.5,min_lr=1e-6)
model = dict(backbone=dict(type="VGG16"), head=dict(type="sal_Decoder"),train_set=train_set)
