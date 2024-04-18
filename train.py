import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from models.augment import augment_image
from models.dataset import build_dataset
import yaml
from models.nets import model_mobilenet_v2
from models.core.loss import JointsMSELoss, JointsMSELoss_offset
from models.core import function, function_offset
from utils import torch_tools, setup_config, file_utils, log
from models.tools.utils import create_work_space
from tensorboardX import SummaryWriter
import torchvision.models as models
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='1'
class Trainer(object):
    def __init__(self, cfg):
        self.is_main_process = True
        torch_tools.set_env_random_seed()  # 来设置随机种子
        self.work_dir = create_work_space(cfg)  # 创建工作目录
        self.model_dir = os.path.join(self.work_dir, "model")
        self.log_dir = os.path.join(self.work_dir, "log")
        if self.is_main_process:
            file_utils.create_dir(self.work_dir)
            file_utils.create_dir(self.model_dir)
            file_utils.create_dir(self.log_dir)
            # file_utils.copy_file_to_dir(config_file, work_dir)
            setup_config.save_config(cfg, os.path.join(self.work_dir,
                                                       "setup_config.yaml"))  # 将配置（cfg）保存到文件setup_config.yaml中。
        # 创建一个日志记录器（self.logger），日志级别为"debug"，日志文件为train.log，是否为主进程由self.is_main_process确定。
        self.logger = log.set_logger(level="debug",
                                     logfile=os.path.join(self.log_dir, "train.log"),
                                     is_main_process=self.is_main_process)

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:0')
        self.train_loader = self.build_train_loader(cfg)  # 创建训练数据加载器
        self.val_loader = self.build_val_loader(cfg)

        self.writer = SummaryWriter(log_dir=self.log_dir)  # 创建一个SummaryWriter对象，用于TensorBoard日志记录，日志目录为log_dir。
        self.writer_dict = {'writer': self.writer, 'train_global_steps': 0, 'valid_global_steps': 0}
        self.target_type = cfg['MODEL']['TARGET_TYPE']
        self.cfg = cfg
        self.build(cfg)  # 调用build函数，根据配置构建模型。
        self.logger.info("=" * 60)  # 打印一系列关于配置和设置的信息
        self.logger.info("work_dir          :{}".format(self.work_dir))
        # self.logger.info("config_file       :{}".format(config_file))
        # self.logger.info("gpu_id            :{}".format(self.gpu_id))
        self.logger.info("main device       :{}".format(self.device))
        self.logger.info("num_samples(train):{}".format(self.num_samples))
        self.logger.info("image size        :{}".format(cfg['MODEL']['IMAGE_SIZE']))
        self.logger.info("target            :{}".format(self.target_type))
        self.logger.info("=" * 60)

    def build_train_loader(self, cfg):
        """ define train dataset"""
        self.logger.info("build_train_loader")
        transform = transforms.Compose([
            transforms.ToPILImage(),
            augment_image.RandomColorJitter(p=0.5,
                                            brightness=0.5,
                                            contrast=0.5,
                                            saturation=0.5,
                                            hue=0.1),
            transforms.RandomChoice([
                augment_image.RandomMotionBlur(degree=5, angle=360, p=0.5),
                augment_image.RandomGaussianBlur(ksize=(1, 1, 1, 3, 3, 5)),
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])# 定义训练数据的数据转换
        dataset = build_dataset.load_dataset(cfg,  # 加载训练数据集
                                             cfg['DATASET']['ROOT'],
                                             cfg['DATASET']['TRAIN_SET'],  # 训练集的名称
                                             True,  # 加载训练集
                                             transform,  # 数据转换函数
                                             True)  # 进行数据集缓存
        loader = torch.utils.data.DataLoader(dataset,  # 构建数据加载器
                                             batch_size=cfg['TEST']['BATCH_SIZE'],
                                             shuffle=cfg['TRAIN']['SHUFFLE'],  # 是否对数据进行随机洗牌
                                             num_workers=cfg['WORKERS'],  # 用于数据加载的工作线程数
                                             pin_memory=True  # 是否将数据加载到 CUDA 的固定内存中
                                             )
        self.num_samples = len(dataset)  # 记录训练样本的数量
        self.logger.info("have train images {}".format(self.num_samples))
        return loader


    def build_val_loader(self, cfg):
        """ define test dataset"""
        self.logger.info("build_val_loader")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = build_dataset.load_dataset(cfg,
                                             cfg['DATASET']['ROOT'],
                                             cfg['DATASET']['TEST_SET'],
                                             False,
                                             transform,
                                             False)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=cfg['TEST']['BATCH_SIZE'],
                                             shuffle=False,
                                             num_workers=cfg['WORKERS'],
                                             pin_memory=True
                                             )
        self.logger.info("have valid images {}".format(len(dataset)))
        return loader

    def build(self, cfg):
        self.model = self.build_model(cfg)
        self.criterion = self.build_criterion(cfg)
        self.optimizer = self.build_optimizer(cfg, self.model)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,  # 学习率调度器
                                                                 cfg['TRAIN']['LR_STEP'],  # 一个列表或元组，表示学习率衰减的步骤
                                                                 cfg['TRAIN']['LR_FACTOR'])  # 衰减因子，用于控制学习率的衰减幅度

    def build_model(self, cfg):
        """define model"""
        self.logger.info("build_model,net_type:{}".format(cfg['MODEL']['NAME']))
        # model = build_nets(net_type=MODEL.NAME, config=cfg, is_train=True)  # 构建模型
        width_mult = cfg['MODEL']['EXTRA']['WIDTH_MULT']
        # model = models.mobilenet_v2(pretrained=True)
        model = model_mobilenet_v2.get_pose_net(cfg, is_train=True, width_mult=width_mult)

        # 加载预训练模型的权重
        pretrained_weights_path = 'pretrained/best_model_178_0.6272.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
        model = model.to(self.device)
        return model

    def build_criterion(self, cfg):
        """define loss function (criterion)"""
        self.logger.info("build_criterion")
        if self.target_type == 'gaussian':  # 目标类型 'gaussian' 表示模型的目标是预测关键点的坐标位置，通常用于姿态估计或关键点检测任务
            criterion = JointsMSELoss(use_target_weight=cfg['LOSS']['USE_TARGET_WEIGHT'])  # 使用MSE计算预测值与真实值之间的均方误差，目标权重（use_target_weight）用于加权不同关键点的重要性
        elif self.target_type == 'offset':  # 目标类型 'offset' 表示模型的目标是预测关键点的偏移量或相对位置，而不是直接预测关键点的绝对坐标位置。其中关键点的位置是相对于人体的主干（如脊柱）或其他参考点的偏移量。
            criterion = JointsMSELoss_offset(use_target_weight=cfg['LOSS']['USE_TARGET_WEIGHT'])
        else:
            raise Exception("Error:{}".format(self.target_type))
        criterion = criterion.to(self.device)  # 将损失函数移动到指定的设备上进行计算，通常是 CUDA 设备
        return criterion

    def build_optimizer(self, cfg, model):
        """ define train optimizer"""
        self.logger.info("build_optimizer")
        self.logger.info("optim_type:{},init_lr:{},weight_decay:{}".format(cfg['TRAIN']['OPTIMIZER'],
                                                                           cfg['TRAIN']['LR'],
                                                                           cfg['TRAIN']['WD']))
        self.logger.info("batch_size:{}".format(cfg['TRAIN']['BATCH_SIZE']))
        if cfg['TRAIN']['OPTIMIZER'] == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=cfg['TRAIN']['LR'],
                                        momentum=cfg['TRAIN']['MOMENTUM'],
                                        weight_decay=cfg['TRAIN']['WD'],
                                        nesterov=cfg['TRAIN']['NESTEROV']
                                        )
        elif cfg['TRAIN']['OPTIMIZER'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['TRAIN']['LR'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.LR, weight_decay=cfg.TRAIN.WD)
        elif cfg['TRAIN']['OPTIMIZER'] == 'adamw':
            # optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WD'])
        else:
            raise Exception("Error:{}".format(cfg['TRAIN']['OPTIMIZER']))
        return optimizer

    def run_train_epoch(self, dataset, epoch):
        # set to training mode
        self.model.train()  # 将模型设置为训练模式
        # 执行实际的训练过程
        self.train(self.cfg, dataset, self.model, self.criterion, self.optimizer, self.writer_dict, epoch, self.device)
        self.lr_scheduler.step()   # 学习率调度器根据当前的 epoch 更新学习率

    def run_val_epoch(self, dataset, epoch):
        # set to evaluates mode
        self.model.eval()
        ap = self.validate(self.cfg, dataset, self.model, self.criterion, self.writer_dict, epoch, self.device)
        return ap

    def run(self, ):
        self.max_ap = 0.0
        if self.target_type == 'offset':
            self.train = function_offset.train
            self.validate = function_offset.validate
        else:
            self.train = function.train
            self.validate = function.validate
        self.logger.info('target: {}'.format(self.target_type))
        for epoch in range(self.cfg['TRAIN']['BEGIN_EPOCH'], self.cfg['TRAIN']['END_EPOCH']):
            # train for one epoch : image is BGR image
            self.logger.info('work space: {}'.format(self.work_dir))
            self.run_train_epoch(self.train_loader, epoch)
            ap = self.run_val_epoch(self.val_loader, epoch)
            self.writer_dict['writer'].add_scalar('lr_epoch', self.optimizer.param_groups[0]['lr'], epoch)
            self.logger.info('=> saving checkpoint to {}'.format(self.model_dir))
            self.logger.info('AP: {}'.format(ap))
            self.save_model(self.model_dir, ap, epoch)

    def save_model(self, model_dir, ap, epoch, start_save=0):
        """
        :param model_dir:
        :param ap:
        :param epoch:
        :param start_save:
        :return:
        """
        model = self.model
        optimizer = self.optimizer
        model_file = os.path.join(model_dir, "model_{:0=3d}_{:.4f}.pth".format(epoch, ap))
        optimizer_pth = os.path.join(model_dir, "model_optimizer.pth")
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, optimizer_pth)

        start_save = start_save if start_save else self.cfg['TRAIN']['END_EPOCH'] - 10
        if epoch >= start_save:
            torch.save(model.state_dict(), model_file)
            self.logger.info("save model in:{}".format(model_file))

        if self.max_ap <= ap:
            self.max_ap = ap
            best_model_file = os.path.join(model_dir, "best_model_{:0=3d}_{:.4f}.pth".format(epoch, ap))
            file_utils.remove_prefix_files(model_dir, "best_model_*")
            torch.save(model.state_dict(), best_model_file)
            self.logger.info("save best_model_path in:{}".format(best_model_file))


if __name__ == '__main__':
    with open('pretrained/train_model_mbv2_penson.yaml', 'r') as f:
        # cfg = yaml.load(f)
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(cfg)
    t = Trainer(cfg)
    t.run()
