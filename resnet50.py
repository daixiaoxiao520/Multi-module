import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd



def draw_features(width,height,x,savename):
    tic=time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        # plt.tight_layout()
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
        # print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    # print("time:{}".format(time.time()-tic))


class ft_net(nn.Module):

    def __init__(self):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft

    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)
            # draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(savepath))
            con_feature = x.cpu().numpy().flatten()
            # pd.DataFrame(con_feature).T.to_csv("{}/f1_conv1_feature.csv".format(savepath),header=False,index=False)

            x = self.model.bn1(x)
            # draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(savepath))
            bn1_feature = x.cpu().numpy().flatten()
            # pd.DataFrame(bn1_feature).T.to_csv("{}/f2_bn1_feature.csv".format(savepath),header=False,index=False)

            x = self.model.relu(x)
            # draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(savepath))
            relu_feature = x.cpu().numpy().flatten()
            # pd.DataFrame(relu_feature).T.to_csv("{}/f3_relu_feature.csv".format(savepath),header=False,index=False)

            x = self.model.maxpool(x)
            # draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(savepath))
            maxpool_feature = x.cpu().numpy().flatten()
            pd.DataFrame(maxpool_feature).T.to_csv("{}/f4_maxpool_feature.csv".format(savepath),header=False,index=False)

            x = self.model.layer1(x)
            # draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(savepath))
            layer1_feature = x.cpu().numpy().flatten()
            # pd.DataFrame(layer1_feature).T.to_csv("{}/f5_layer1_feature.csv".format(savepath),header=False,index=False)

            x = self.model.layer2(x)
            # draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(savepath))
            layer2_feature = x.cpu().numpy().flatten()
            # pd.DataFrame(layer2_feature).T.to_csv("{}/f6_layer2_feature.csv".format(savepath),header=False,index=False)

            x = self.model.layer3(x)
            # draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(savepath))
            layer3_feature = x.cpu().numpy().flatten()
            pd.DataFrame(layer3_feature).T.to_csv("{}/f7_layer3_feature.csv".format(savepath),header=False,index=False)

            x = self.model.layer4(x)
            # draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(savepath))
            # draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(savepath))
            layer4_feature = x.cpu().numpy().flatten()
            pd.DataFrame(layer4_feature).T.to_csv("{}/f8_layer4_feature.csv".format(savepath),header=False,index=False)


            # x = self.model.avgpool(x)
            # plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            # plt.savefig("{}/f9_avgpool.png".format(savepath))
            # plt.clf()
            # plt.close()

            # x = x.view(x.size(0), -1)
            # x = self.model.fc(x)
            # plt.plot(np.linspace(1, 1000, 1000), x.cpu().numpy()[0, :])
            # plt.savefig("{}/f10_fc.png".format(savepath))
            # plt.clf()
            # plt.close()
        else :
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.fc(x)

        return x


model=ft_net().cuda()

# pretrained_dict = resnet50.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# net.load_state_dict(model_dict)
model.eval()
save_path=r'E:\multimodal ultrasonography\712all_images\214_test_data\ZQM_mask\crop_C_feature'
rootpath = r'E:\multimodal ultrasonography\712all_images\214_test_data\ZQM_mask\crop_C'
images = os.listdir(rootpath)
for image in images:
    if True:
        img=cv2.imread(os.path.join(rootpath, image))
        savepath = os.path.join(save_path, image.split('.')[0])
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        img=cv2.resize(img,(224,224));
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        img=transform(img).cuda()
        img=img.unsqueeze(0)
        with torch.no_grad():
            start=time.time()
            out=model(img)
            print("total time:{}".format(time.time()-start))
            result=out.cpu().numpy()
            # ind=np.argmax(out.cpu().numpy())
            ind=np.argsort(result,axis=1)
            # for i in range(5):
            #     print("predict:top {} = cls {} : score {}".format(i+1,ind[0,1000-i-1],result[0,1000-i-1]))
            # print("done")
