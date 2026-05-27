import os
os.environ["WANDB_API_KEY"] = '3d0f14304695197a773e59b027afa3b3c4ca46e1'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torch import optim
import matplotlib.pyplot as plt
import matplotlib
import time
import metrics
from MyDataset import MyDataset
import datetime
import numpy as np
# from deeplabv3plus_res18 import DeepLabV3Plus
from deeplabv3plus_res101 import DeepLabV3Plus
from U_net import Unet
from FCN import FCN8s
from PSPNet import PSPNet
# from swinunet import SwinUnet
from seg_swinv2unet import SwinUnet,Swin_LCSRB_PSP_FPNPAN,Swin_LCSRB,Swin_FPNPAN,Swin_PSP,Swin_DeformablePSP,Swin_LCSRB_PSP,Swin_LCSRB_FPNPAN,Swin_FPNPAN_PSP,Swin_LCSRB_DeformablePSP_FPNPAN
from transunet import TransUNet
from seg_swinv2unet import Swin_FPNPAN_DeformablePSP,Swin_LCSRB_DeformablePSP



def showTensor(data, title="data"):
    data = data.cpu()

    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(5, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(data.detach().numpy(), cmap='gray')
    plt.title(title)

    plt.show()
    plt.close(fig)

def denormalize(tensor_image):
    unnormalized_tensor = tensor_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

    # 对ToTensor操作进行反操作
    numpy_image = unnormalized_tensor.numpy()
    numpy_image = numpy_image.transpose((1, 2, 0))
    numpy_image = (numpy_image * 255).astype(np.uint8)

    # 将numpy.ndarray转换为PIL图像
    # restored_image = Image.fromarray(numpy_image)
    return numpy_image

def savePredictResult(optical, label, y_hat, pre, save_path, title):
    optical = optical.cpu()#3,512,512
    #这里作反归一化
    optical = denormalize(optical)

    label = label.cpu()
    y_hat = y_hat.cpu()
    pre = pre.cpu()
    # matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(optical, cmap='gray')
    plt.title('optical')

    plt.subplot(1, 3, 2)
    plt.imshow(label.detach().numpy(), cmap='gray')
    plt.title('label')
    plt.subplot(1, 3, 3)
    plt.imshow(pre.detach().numpy(), cmap='gray')
    plt.title('predict')
    img_y = torchvision.transforms.ToPILImage()(y_hat)
    img_predict = torchvision.transforms.ToPILImage()(pre)
    # save_path = os.path.join(save_path, title)
    if os.path.exists(save_path) is not True:
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, title + '.png'))
    img_y.save(os.path.join(save_path, title + '_y.png'), 'PNG')
    img_predict.save(os.path.join(save_path, title + '_pre.png'), 'PNG')
    plt.close(fig)


def net_test(model, test_iter, loss, record_path, epoch='000',save=False):
    model.eval()
    test_epoch_loss = []
    TP_total = FP_total = TN_total = FN_total = 0
    for optical, label, img_name in tqdm(test_iter, desc=f'test     Epoch {epoch}    :', unit='img'):
        optical, label = optical.to(device), label.to(device)
        # y_hat = model(aspect, dem)
        y_hat = model(optical)
        # print('y_hat',y_hat.shape)
        y_hat = nn.Sigmoid()(y_hat)
        # print('sigmoid y_hat',y_hat.shape)
        # print(aspect)
        # 计算损失
        l = loss(y_hat, label)
        test_loss = l.item()
        test_epoch_loss.append(test_loss)
        pred = (y_hat > 0.45).to(torch.float32)
        TP, TN, FP, FN = metrics.confusion_matrix(pred, label)
        TP_total += TP
        TN_total += TN
        FP_total += FP
        FN_total += FN
        test_acc = metrics.accuracy(TP, TN, FP, FN)
        try:
            test_precision = metrics.precision(TP, FP)
            test_recall = metrics.recall(TP, FN)
            test_f1 = metrics.f1_score(test_precision, test_recall)
            test_iou = metrics.iou_score(TP, FP, FN)
        except:
            test_precision = test_recall = test_f1 = test_iou = 0

        # print(
        #     '\ntest_loss: {:.10f}  test_acc: {:.10f}  test_precision: {:.10f}  test_recall: {:.10f}  test_f1: {:.10f}  test_iou: {:.10f}   {}'.format(
        #         test_loss, test_acc, test_precision, test_recall, test_f1, test_iou, img_name[0]))

        # wandb.log(
        #     {'test_loss': test_loss, 'test_acc': test_acc, 'test_precision': test_precision, 'test_recall': test_recall,
        #      'test_f1': test_f1, 'test_iou': test_iou})
        if save:
            title = 'test_{}_{}'.format(img_name[0], str(int(time.time() * 100)))
            # savePredictResult(optical[0].reshape([512, 512]), label[0].reshape([512, 512]),
            #                 y_hat[0].reshape([512, 512]),
            #                 pred[0].reshape([512, 512]),
            #                 os.path.join(record_path ,f'test_{epoch}'), title)
            savePredictResult(optical[0], label[0][0],
                            y_hat[0].reshape([512, 512]),
                            pred[0].reshape([512, 512]),
                            os.path.join(record_path ,f'test_{epoch}'), title)
    test_epoch_loss = np.average(test_epoch_loss)
    try:
        test_acc = metrics.accuracy(TP_total, TN_total, FP_total, FN_total)
        test_precision = metrics.precision(TP_total, FP_total)
        test_recall = metrics.recall(TP_total, FN_total)
        test_f1 = metrics.f1_score(test_precision, test_recall)
        test_iou = metrics.iou_score(TP_total, FP_total, FN_total)
    except:
        test_acc = test_precision = test_recall = test_f1 = test_iou = 0.0
    print('- ' * 30)
    print('\nTP:{}   TN:{}   FP:{}   FN:{}'.format(TP_total, TN_total, FP_total, FN_total))
    print(
        '\ntest_loss: {:.10f}  test_acc: {:.10f}  test_precision: {:.10f}  test_recall: {:.10f}  test_f1: {:.10f}  test_iou: {:.10f}'.format(
            test_epoch_loss, test_acc, test_precision, test_recall, test_f1, test_iou))
    print('- ' * 30)
    # wandb.log(
    #     {'test_loss': test_epoch_loss, 'test_acc': test_acc, 'test_precision': test_precision,
    #      'test_recall': test_recall,
    #      'test_f1': test_f1, 'test_iou': test_iou})
    return test_iou


def train(model, train_iter, val_iter, loss, opt, num_epochs, record_path, lr_scheduler):
    """    test_table = wandb.Table(data=None, columns=['id', 'epoch', 'acc'])
    """
    test_iou = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        # eval_data = dict({'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': [], 'iou': []})
        train_epoch_loss = []
        TP_total = FP_total = TN_total = FN_total = 0
        for optical, label, img_name in tqdm(train_iter, desc=f'Epoch {epoch}/{num_epochs} ', unit='img'):
            # 正向传播
            optical, label = optical.to(device), label.to(device)
            # showTensor(label[0][0], title='label')
            # y_hat = model(aspect, dem)
            # print(optical.shape)
            y_hat = model(optical)

            y_hat = nn.Sigmoid()(y_hat)

            # print(aspect)
            # 计算损失
            l = loss(y_hat, label)

            # 反向传播
            opt.zero_grad()
            l.backward()

            # 优化
            opt.step()
            train_loss = l.item()
            train_epoch_loss.append(train_loss)
            # tensor.item() 将单元素tensor转化为python标量
            pred = (y_hat > 0.45).to(torch.float32)

            TP, TN, FP, FN = metrics.confusion_matrix(pred, label)

            TP_total += TP
            TN_total += TN
            FP_total += FP
            FN_total += FN

            # eval_data['loss'].append(train_loss), eval_data['acc'].append(train_acc), eval_data['precision'].append(
            try:
                train_precision = metrics.precision(TP, FP)
                train_recall = metrics.recall(TP, FN)
                train_f1 = metrics.f1_score(train_precision, train_recall)
                train_iou = metrics.iou_score(TP, FP, FN)
            except:
                train_precision = train_recall = train_f1 = train_iou = 0
            train_acc = metrics.accuracy(TP, TN, FP, FN)
            # print(TP, TN, FP, FN, (pred == label).sum(), label.numel())
            #     train_precision), eval_data['recall'].append(train_recall), eval_data['f1'].append(train_f1), eval_data[
            #     'iou'].append(train_iou)
            # print(
            #     '\ntrain_loss:{:.10f}  train_acc:{:.10f}  train_precision:{:.10f}  train_recall:{:.10f}  train_f1:{:.10f}  train_iou:{:.10f}  lr:{}  img_name:{}'.format(
            #         train_loss, train_acc, train_precision, train_recall, train_f1, train_iou,
            #         lr_scheduler.get_last_lr(), str(img_name)))
            # print(f'\nTP:{TP}    TN:{TN}    FP:{FP}    FN:{FN}')
            # wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'train_precision': train_precision,
            #            'train_recall': train_recall, 'train_f1': train_f1, 'train_iou': train_iou})

            # if epoch in [15, 25,40] and train_acc != 1.0:
                #test_table.add_data(str(img_name), epoch, train_acc)
                # wandb.log({"img_table":test_table})
                # if epoch>=0 and train_acc<0.9:
                #     wandb.log({'image_name':img_name,'aspect0':wandb.Image(aspect.cpu()[0].detach().numpy()),'aspect1':wandb.Image(aspect.cpu()[1].detach().numpy()),
                #                'dem0':wandb.Image(dem.cpu()[0].detach().numpy()),'dem1':wandb.Image(dem.cpu()[1].detach().numpy()),
                #                'label0':wandb.Image(label.cpu()[0].detach().numpy()),'label1':wandb.Image(label.cpu()[1].detach().numpy())})
                #     print('wandb image')
                # for img_index in range(0, len(img_name)):
                #     title = 'epoch{}_{}_{}'.format(epoch, img_name[img_index], str(int(time.time() * 100)))
                #     savePredictResult(optical[img_index].reshape([512, 512]),
                #                       label[img_index].reshape([512, 512]),
                #                       y_hat[img_index].reshape([512, 512]),
                #                       pred[img_index].reshape([512, 512]),
                #                       record_path + r'\epoch' + str(epoch), title)

                # title = 'epoch{}_{}_{}'.format(epoch, img_name[1], str(int(time.time() * 100)))
                # savePredictResult(dem[1].reshape([512, 512]), aspect[1].reshape([512, 512]),
                #                   label[1].reshape([512, 512]),
                #                   y_hat[1].reshape([512, 512]),
                #                   pred[1].reshape([512, 512]),
                #                   record_path + r'\epoch' + str(epoch), title)

        train_acc = metrics.accuracy(TP_total, TN_total, FP_total, FN_total)
        train_precision = metrics.precision(TP_total, FP_total)
        train_recall = metrics.recall(TP_total, FN_total)
        train_f1 = metrics.f1_score(train_precision, train_recall)
        train_iou = metrics.iou_score(TP_total, FP_total, FN_total)
        train_epoch_loss = np.average(train_epoch_loss)
        print('- ' * 30)
        print(
            'train_loss:{:.10f}  train_acc:{:.10f}  train_precision:{:.10f}  train_recall:{:.10f}  train_f1:{:.10f}  train_iou:{:.10f}'.format(
                train_epoch_loss, train_acc, train_precision, train_recall, train_f1, train_iou))
        # print(np.average(eval_data['loss']),np.average(eval_data['acc']),np.average(eval_data['precision']),np.average(eval_data['recall']),np.average(eval_data['f1']),np.average(eval_data['iou']))
        print('- ' * 30)
        """wandb.log({'train_loss': train_epoch_loss, 'train_acc': train_acc, 'train_precision': train_precision,
                   'train_recall': train_recall, 'train_f1': train_f1, 'train_iou': train_iou})
        """
        lr_scheduler.step()  # 更新学习率
        test_epoch_iou = net_test(model=model, test_iter=test_iter, loss=loss, record_path=record_path, epoch=str(epoch),save=False)

        #保存最优模型
        if test_epoch_iou > test_iou:
            test_iou = test_epoch_iou
            torch.save(model, hp.model_save_path)
            print(f'save best model at epoch {epoch}')


    # wandb.log({'f1_list': test_table})


if __name__ == '__main__':
    class HyperParameter:
        def __init__(self):
            curr_time = datetime.datetime.now()
            curr_time_str = curr_time.strftime("_%Y%m%d_%H%M%S")
            self.name = "_result" + curr_time_str
            self.num_epochs = 50  # 模型跑了多少轮
            self.learning_rate = 6e-5  # 学习率
            self.train_batchsize = 4  # 一次多少张图片
            self.test_batchsize = 1
            # self.val_save_path = r'/model_unet/datasave/temp/val12'
            self.train_dataset_path = r'F:\lunar_lobate_scraps\Experment_6\train8_grey_1'  # train8_grey
            self.test_dataset_path = r'F:\lunar_lobate_scraps\Experment_6\test2_grey_1'  # test2_grey
            self.record_path = r'F:\lunar_lobate_scraps\Experment_6\TransUNet_ep50_rate000001_batch4'
            self.model_save_path = os.path.join(self.record_path, self.name + '.pt')
            self.label_weight = 5
            self.low_drop = 0
            self.high_drop = 0  # 0.2
            self.backbone_drop = 0  # 0.1


    hp = HyperParameter()
    # HyperParameter = {
    #     'train batchsize': 64,
    #     'test batchsize': 10,
    #     'learning rate': 0.1,
    #     'epochs': 3

    # }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}     GPU available:{torch.cuda.is_available()}')
    if torch.cuda.is_available() is not True:
        print('GPU  not available')
        exit()

    # model = DeepLabV3Plus(1).to(device)
    model = TransUNet().to(device)

    #model = SwinUnet().to(device)
    #model = Swin_LCSRB().to(device)
    #model = Swin_FPNPAN().to(device)
    #model = Swin_DeformablePSP().to(device)

    # model = Swin_LCSRB_PSP().to(device)
    # model = Swin_LCSRB_FPNPAN().to(device)
    # model = Swin_FPNPAN_PSP().to(device)
    # model = Swin_LCSRB_PSP_FPNPAN().to(device)

    #model = Swin_FPNPAN_DeformablePSP().to(device)
    #model = Swin_LCSRB_DeformablePSP().to(device)
    #model=Swin_LCSRB_FPNPAN().to(device)

    model = Swin_LCSRB_DeformablePSP_FPNPAN().to(device)
    ckpt = torch.load(r'F:\lunar_lobate_scraps\seg\Swin_LCSRB_DeformablePSP_FPNPAN_ep50_rate000001_batch4\_result_20241126_214348.pt')
    model.load_state_dict(ckpt.state_dict())



    

    transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = MyDataset(os.path.join(hp.train_dataset_path,r'F:\lunar_lobate_scraps\Experment_6\train8_grey_1'),        #train8_grey
                        os.path.join(hp.train_dataset_path,r'F:\lunar_lobate_scraps\Experment_6\train8_label_1'),       #train8_label_
                        transform)

    train_iter = DataLoader(dataset=train_data, batch_size=hp.train_batchsize, shuffle=True,
                            drop_last=False)  # ,num_workers=4,prefetch_factor=2

    test_data = MyDataset(os.path.join(hp.test_dataset_path,r'F:\lunar_lobate_scraps\Experment_6\test2_grey_1'),           #test2_grey
                        os.path.join(hp.test_dataset_path,r'F:\lunar_lobate_scraps\Experment_6\test2_label_1'),          #test2_label
                        transform)

    test_iter = DataLoader(dataset=test_data, batch_size=hp.test_batchsize, shuffle=False, drop_last=True)

    # loss = torch.nn.BCELoss(weight=torch.tensor([1.5]))
    # loss = torch.nn.BCEWithLogitsLoss()
    loss = torch.nn.BCELoss(weight=torch.FloatTensor([hp.label_weight])).to(device)

    # opt = optim.SGD(model.parameters(), lr=hp.learning_rate,weight_decay=1e-3)
    # opt = optim.SGD(model.parameters(), lr=hp.learning_rate)
    opt = optim.AdamW(model.parameters(), lr=hp.learning_rate)

    
    test_epoch_iou = net_test(model=model, test_iter=test_iter, loss=loss, record_path=hp.record_path, epoch=50,save=True)
    print(test_epoch_iou)