# _base_=[
#        '../_base_/models/encoder_decoder/convnext_large.hourglassdecoder.py',
#        '../_base_/datasets/_data_base_.py',
#        '../_base_/default_runtime.py',
#        ]

model = dict(
    type='EncoderDecoderModel',
    backone=dict(
        type='vgg16',
        prefix='backbones.',
        pretrained=True,
        out_channels=25088,
    ),
    decode_head=dict(
        type='SimpleDecoder',
        prefix='decode_heads.',
        out_channels=10,
    ),
)


batchsize_per_gpu = 2
thread_per_gpu = 4



init_lr = dict(
    encoder = 0,
    decoder = 0,
)

optimizer = dict(
    algorithms = "SGD",
    lr = 0,
    momentum = 0.9,
)

train_data = [
    dict(AVA="AVA_dataset"),

]

AVA_dataset = dict(
    lib = 'AVADataset',
    data_type = '',
    train = dict(
        sample_ration = 1.0,
        sample_size = -1,
        pipeline=[
          dict(type='BGR2RGB'),
          dict(type='ResizeKeepRatio',
               resize_size=[256,256],),
          dict(type='RandomCrop',
               crop_size=[224,224],),
          dict(type='RandomHorizontalFlip',
               probability=0.5,),
          dict(type='Normalize',
               mean=[0.485, 0.456, 0.406], 
               std=[0.229, 0.224, 0.225],),
          dict(type='ToTensor',),
        ]
    )
)