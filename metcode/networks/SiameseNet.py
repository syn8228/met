import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import sys
sys.path.append('/cluster/yinan/met/')

from metcode.networks.backbone import Embedder


class ResNet50Conv4(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Conv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.features(x)
        return x


class VGG16Pool5(nn.Module):
    def __init__(self):
        super(VGG16Pool5, self).__init__()
        self.net = torchvision.models.vgg16(pretrained=True).features

    def forward(self, x):
        x = self.net(x)
        return x


class VGG16FC6(nn.Module):
    def __init__(self):
        super(VGG16FC6, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.avgpool = torchvision.models.vgg16(pretrained=True).avgpool
        self.classifier = torchvision.models.vgg16(pretrained=True).classifier[0]
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class VGG16FC7(nn.Module):
    def __init__(self):
        super(VGG16FC7, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.avgpool = torchvision.models.vgg16(pretrained=True).avgpool
        self.classifier = torchvision.models.vgg16(pretrained=True).classifier[:4]
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def load_siamese_checkpoint(name, checkpoint_file):
    if name == "resnet50":
        print('--------------------------------------------------------------')
        print('used model: zoo_resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=True)
        # model.eval()
        return model

    elif name == "vgg_fc7":
        print('--------------------------------------------------------------')
        print('used model: VGG16_fc7')
        print('--------------------------------------------------------------')
        model = VGG16FC7()
        # model.eval()
        return model

    # TODO: Train from scratch if the network weights are not available
    else:
        print('--------------------------------------------------------------')
        print('used model: resnet50')
        print('--------------------------------------------------------------')
        model = torchvision.models.resnet50(pretrained=False)
        model.eval()
        return model


class siamese_network(nn.Module):
	'''Network architecture for contrastive learning.
	'''

	def __init__(self,backbone,pooling = "gem",pretrained = True,
					emb_proj = False,init_emb_projector= None):

		super(siamese_network,self).__init__()

		net = Embedder(backbone,gem_p = 3.0,pretrained_flag = pretrained,
						projector = emb_proj,init_projector = init_emb_projector) 

		self.backbone = net	#the backbone produces l2 normalized descriptors


	def forward(self,augs1,augs2):

		descriptors_left = self.backbone(augs1)
		descriptors_right = self.backbone(augs2)

		return descriptors_left,descriptors_right


class TripletSiameseNetwork_custom(nn.Module):
    def __init__(self, model, checkpoint='/cluster/yinan/isc2021/data/multigrain_joint_3B_0.5.pth'):
        super(TripletSiameseNetwork_custom, self).__init__()
        self.model = model
        self.head = load_siamese_checkpoint(model, checkpoint)
        self.flatten = nn.Flatten()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.clamp(x, eps, np.inf)
        x = x ** p
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x ** (1. / p)

    def forward_once(self, x):
        if self.model == 'resnet50':
            x = self.head.conv1(x)
            x = self.head.bn1(x)
            x = self.head.relu(x)
            x = self.head.maxpool(x)

            x1 = self.head.layer1(x)
            x2 = self.head.layer2(x1)
            x3 = self.head.layer3(x2)
            x4 = self.head.layer4(x3)
            x1 = self.gem(x1)
            x1 = self.flatten(x1)
            x2 = self.gem(x2)
            x2 = self.flatten(x2)
            x3 = self.gem(x3)
            x3 = self.flatten(x3)
            x4 = self.gem(x4)
            x4 = self.flatten(x4)
            return x1, x2, x3, x4

        elif self.model == 'vgg_fc7':
            '''relu1_2'''
            x1 = self.head.features[:3](x)
            '''relu2_2'''
            x2 = self.head.features[4:9](x1)
            '''relu3_3'''
            x3 = self.head.features[9:16](x2)
            '''relu4_3'''
            x4 = self.head.features[16:23](x3)
            '''linear classifier'''
            x5 = self.head.features[23:](x4)
            x5 = self.head.avgpool(x5)
            x5 = self.flatten(x5)
            x5 = self.head.classifier[0](x5)
            x6 = self.head.classifier[1:](x5)
            x5 = F.normalize(x5)
            # x6 = F.normalize(x6)

            x1 = F.adaptive_max_pool2d(x1, (1, 1))
            x1 = self.flatten(x1)
            x1 = F.normalize(x1)
            x2 = F.adaptive_max_pool2d(x2, (1, 1))
            x2 = self.flatten(x2)
            x2 = F.normalize(x2)
            x3 = F.adaptive_max_pool2d(x3, (1, 1))
            x3 = self.flatten(x3)
            x3 = F.normalize(x3)
            x4 = F.adaptive_max_pool2d(x4, (1, 1))
            x4 = self.flatten(x4)
            x4 = F.normalize(x4)
            return x1, x2, x3, x4, x5, x6

    def forward(self, input1, input2, input3):
        if self.model == 'resnet50':
            x1_1, x1_2, x1_3, x1_4 = self.forward_once(input1)
            x2_1, x2_2, x2_3, x2_4 = self.forward_once(input2)
            x3_1, x3_2, x3_3, x3_4 = self.forward_once(input3)

            return x1_1, x1_2, x1_3, x1_4, x2_1, x2_2, x2_3, x2_4, x3_1, x3_2, x3_3, x3_4

        elif self.model == 'vgg' or self.model == 'vgg_fc7':
            x1_1, x1_2, x1_3, x1_4, x1_5, x1_6 = self.forward_once(input1)
            x2_1, x2_2, x2_3, x2_4, x2_5, x2_6 = self.forward_once(input2)
            x3_1, x3_2, x3_3, x3_4, x3_5, x3_6 = self.forward_once(input3)

            return x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x3_1, x3_2, x3_3, x3_4, x3_5, x3_6