import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def get_model(model_name, num_classes, log_softmax=False, softmax=True, freeze_params=False, device='cuda:0'):
    if model_name == "DenseNet201":
        model = DenseNet201(weights=models.DenseNet201_Weights.DEFAULT,
                            num_classes=num_classes).to(device)
    elif model_name == "ResNet50":
        model = ResNet50(weights=models.ResNet50_Weights.DEFAULT,
                         num_classes=num_classes,
                         log_softmax=log_softmax,
                         softmax=softmax).to(device)
    elif model_name == "InceptionV3":
        model = InceptionV3(weights=models.Inception_V3_Weights.DEFAULT, num_classes=num_classes).to(device)
    elif model_name == "ResNet18":
        model = ResNet18(weights=models.ResNet18_Weights.DEFAULT,
                         num_classes=num_classes,
                         log_softmax=log_softmax,
                         softmax=softmax).to(device)
    elif model_name == "MobileNetV3Small":
        model = MobileNetV3Small(weights=models.MobileNet_V3_Small_Weights.DEFAULT,
                                 num_classes=num_classes,
                                 log_softmax=log_softmax,
                                 softmax=softmax).to(device)
    elif model_name == "MobileNetV3Large":
        model = MobileNetV3Large(weights=models.MobileNet_V3_Large_Weights.DEFAULT,
                                 freeze_params=freeze_params,
                                 num_classes=num_classes,
                                 log_softmax=log_softmax,
                                 softmax=softmax).to(device)
    elif model_name == "SqueezeNet1_1":
        model = SqueezeNet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT,
                              num_classes=num_classes,
                              log_softmax=log_softmax,
                              softmax=softmax).to(device)
    else:
        raise ValueError('Select a model.')

    return model


def conv_output_size(in_size, kernel, stride=1, padding=0):
    return (in_size - kernel + 2 * padding) // stride + 1


def stable_softmax(logits, dim=1):
    # Subtract the maximum value in each row (or batch) for numerical stability
    max_vals, _ = torch.max(logits, dim=dim, keepdim=True)
    # Subtract max_vals from logits and apply softmax
    exp_vals = torch.exp(logits - max_vals)
    softmax = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
    return softmax


class Audio1DDevIdentification(nn.Module):

    def __init__(self,
                 signal_dim,
                 n_filters=8,
                 input_linear_dim=100,
                 n_linear_out_layer=2,
                 n_cnn_layers=3,
                 kernel_sizes=None,
                 conv_bias=False,
                 linear_bias=False,
                 n_classes=35):
        super().__init__()
        self.n_linear_out_layer = n_linear_out_layer
        self.n_cnn_layers = n_cnn_layers
        self.num_filters = n_filters
        self.in_features = signal_dim
        self.linear_bias = linear_bias
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        self.n_classes = n_classes

        if len(self.kernel_sizes) == 1:
            self.kernel_sizes = n_cnn_layers * kernel_sizes

        assert len(self.kernel_sizes) == n_cnn_layers, 'number of kernels should be equal to the number to conv layers'

        self.linear_in_layer = nn.Sequential(
            nn.Linear(signal_dim, input_linear_dim, bias=False),
            # nn.BatchNorm1d(input_linear_dim),
            # nn.Dropout(p=0.5)
        )

        conv_stride = 1
        max_pool_kernel_size = 2
        max_pool_kernel_size_stride = 2

        conv_layers_out_dim = input_linear_dim
        mod_cnn = []
        for n in range(self.n_cnn_layers):
            in_filters = n_filters if n > 0 else 1

            kernel_size = self.kernel_sizes[n]
            conv_padding = (kernel_size - 1) // 2

            mod_cnn += [
                nn.Conv1d(in_channels=in_filters, out_channels=n_filters, kernel_size=kernel_size,
                          stride=conv_stride, padding=conv_padding, bias=self.conv_bias),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=max_pool_kernel_size_stride),
            ]
            conv_layers_out_dim = conv_output_size(conv_layers_out_dim, kernel_size,
                                                   stride=conv_stride,
                                                   padding=conv_padding)
            conv_layers_out_dim = int((conv_layers_out_dim - max_pool_kernel_size) / max_pool_kernel_size_stride) + 1

        self.hidden_conv_layers = nn.Sequential(*mod_cnn)

        mod_linear = []
        for n in range(self.n_linear_out_layer):
            mod_linear += [
                # nn.BatchNorm1d(conv_layers_out_dim * n_filters),
                nn.Linear(conv_layers_out_dim * n_filters, conv_layers_out_dim * n_filters, bias=self.linear_bias),
                nn.BatchNorm1d(conv_layers_out_dim * n_filters),
                # nn.Tanh(),

                # nn.ReLU(),
                # nn.Dropout(p=0.2),
            ]
        # mod_linear += [nn.Linear(conv_layers_out_dim * n_filters, out_features=self.n_classes, bias=True),
        #                nn.LogSoftmax(dim=1)]
        self.linear_out_layers = nn.Sequential(*mod_linear)

        self.out_layer = nn.Sequential(
            nn.Linear(conv_layers_out_dim * n_filters, out_features=self.n_classes, bias=True),
            nn.LogSoftmax(dim=1)
            # nn.BatchNorm1d(input_linear_dim),
            # nn.Dropout(p=0.5)
        )

    def forward(self, inp):

        bsz = inp.size(0)
        inp = inp.view(bsz, -1)

        x = self.linear_in_layer(inp).view(bsz, 1, -1)

        x = self.hidden_conv_layers(x).view(bsz, -1)

        # intermediate linear layers
        x = self.linear_out_layers(x)
        output = self.out_layer(x)

        return output


class SqueezeNet1_1(nn.Module):
    # https://github.com/culv/SqueezeTune/blob/master/finetune.py
    def __init__(self, num_classes, weights=models.SqueezeNet1_1_Weights.DEFAULT,
                 freeze=False, log_softmax=False, softmax=False):
        super(SqueezeNet1_1, self).__init__()
        self.model = models.squeezenet1_1(weights=weights)
        # Reshape classification layer to have 'num_classes' outputs
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.num_classes = num_classes

        # self.linear_in_layer = nn.Sequential(
        #     nn.Dropout(p=0.5, inplace=False),
        #     nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
        #     nn.Linear(signal_dim, input_linear_dim, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )

        # Replace dropout layer in classifier with batch normalization
        self.model.classifier[0] = nn.BatchNorm2d(512)

        self.log_softmax = log_softmax
        self.softmax = softmax

        if freeze:
            # Freeze all parameters
            for p in self.model.parameters():
                p.requires_grad = False

            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes, weights=models.MobileNet_V3_Large_Weights.DEFAULT,
                 freeze_params=False, log_softmax=False, softmax=False):
        super(MobileNetV3Large, self).__init__()
        self.model = models.mobilenet_v3_large(weights=weights)
        self.model.classifier = nn.Sequential(*[nn.Linear(960, 350),
                                                nn.Linear(350, num_classes)])
        # self.model.classifier = nn.Linear(960, num_classes)

        if freeze_params:
            # Freeze all parameters
            for p in self.model.parameters():
                p.requires_grad = False

            for p in self.model.classifier.parameters():
                p.requires_grad = True

        self.log_softmax = log_softmax
        self.softmax = softmax

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes, weights=models.MobileNet_V3_Small_Weights.DEFAULT,
                 log_softmax=False, softmax=False):
        super(MobileNetV3Small, self).__init__()
        self.model = models.mobilenet_v3_small(weights=weights)

        # num_ftrs = self.model.classifier[-1].in_features
        # self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
        self.model.classifier = nn.Linear(576, num_classes)

        self.log_softmax = log_softmax
        self.softmax = softmax

        assert not (log_softmax and softmax), 'log_softmax and softmax can not be enabled simultaneously'

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class DenseNet201(nn.Module):
    def __init__(self, num_classes, weights=models.DenseNet201_Weights.DEFAULT, log_softmax=False, softmax=False):
        super(DenseNet201, self).__init__()
        self.model = models.densenet201(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
        self.log_softmax = log_softmax
        self.softmax = softmax

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class ResNet50(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet50_Weights.DEFAULT,
                 log_softmax=False,
                 softmax=False):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.log_softmax = log_softmax
        self.softmax = softmax

        assert not (log_softmax and softmax), 'log_softmax and softmax can not be enabled simultaneously'

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class ResNet101(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet101_Weights.DEFAULT,
                 log_softmax=False,
                 softmax=False):
        super(ResNet101, self).__init__()
        self.model = models.resnet101(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.log_softmax = log_softmax
        self.softmax = softmax

        assert not (log_softmax and softmax), 'log_softmax and softmax can not be enabled simultaneously'

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class vit_b_32(nn.Module):
    def __init__(self, num_classes, weights=models.ViT_B_32_Weights.DEFAULT,
                 log_softmax=False, softmax=False):
        super(vit_b_32, self).__init__()

        self.model = models.vit_b_32(weights=weights)
        # modify the classification head
        # setup for two class classification
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, num_classes)

        self.log_softmax = log_softmax
        self.softmax = softmax

        assert not (log_softmax and softmax), 'log_softmax and softmax can not be enabled simultaneously'

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class vit_b_16(nn.Module):
    def __init__(self, num_classes, weights=models.ViT_B_16_Weights.DEFAULT,
                 log_softmax=False, softmax=False):
        super(vit_b_16, self).__init__()

        self.model = models.vit_b_16(weights=weights)
        # modify the classification head
        # setup for two class classification
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, num_classes)

        self.log_softmax = log_softmax
        self.softmax = softmax

        assert not (log_softmax and softmax), 'log_softmax and softmax can not be enabled simultaneously'

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class vit_l_16(nn.Module):
    def __init__(self, num_classes, weights=models.ViT_L_16_Weights.DEFAULT,
                 log_softmax=False, softmax=False):
        super(vit_l_16, self).__init__()

        self.model = models.vit_l_16(weights=weights)
        # modify the classification head
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_ftrs, num_classes)

        self.log_softmax = log_softmax
        self.softmax = softmax

        assert not (log_softmax and softmax), 'log_softmax and softmax can not be enabled simultaneously'

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class ResNet18(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet18_Weights.DEFAULT,
                 log_softmax=False, softmax=False):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)
        self.log_softmax = log_softmax
        self.softmax = softmax

        assert not (log_softmax and softmax), 'log_softmax and softmax can not be enabled simultaneously'

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        elif self.softmax:
            # return F.softmax(output, dim=1)
            return stable_softmax(output, dim=1)
        else:
            return output


class InceptionV3(nn.Module):
    def __init__(self, num_classes, weights=models.Inception_V3_Weights.DEFAULT, log_softmax=False):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(weights=weights, aux_logits=False)
        self.model.fc = nn.Linear(2048, num_classes)
        self.log_softmax = log_softmax

    def forward(self, x):
        output = self.model(x)
        if self.log_softmax:
            return F.log_softmax(output, dim=1)
        else:
            return output
