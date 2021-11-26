import torch.nn as nn
import torch.nn.functional as F
import torch
import random

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = F.interpolate(x, skip_input.shape[-2:])
        x = torch.cat((x, skip_input), 1)

        return x


class KeypointNet(nn.Module):
    def __init__(self, key_num=5, clutter_num=5):
        super(KeypointNet, self).__init__()

        self.feature_left = FeatureExtractor()
        self.feature_right = FeatureExtractor()
        self.representation = Representation(clutter_num=clutter_num)
        self.generator1 = Generator1(key_num=key_num)
        self.generator2 = Generator2(key_num=key_num)

    def forward(self, left, right, uv_list=None, mode="train"):

        feature_map_l  = self.feature_left(left)
        feature_map_r  = self.feature_right(right)

        feature_map = torch.cat((feature_map_l,feature_map_r), axis=1)

        middle_heat_l, middle_heat_r, rgb, depth = self.generator1(feature_map)
        final_heat_l, final_heat_r = self.generator2(middle_heat_l, middle_heat_r, rgb, depth)

        if mode == "train":
            key_vector, clutter_vector = self.representation(feature_map, uv_list)                     
            return key_vector, clutter_vector, middle_heat_l, final_heat_l, middle_heat_r, final_heat_r, rgb, depth
        
        elif mode == "val":
            return middle_heat_l, final_heat_l, middle_heat_r, final_heat_r, rgb, depth

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 1024, dropout=0.5)

        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 256, 4, padding=1),
            nn.LeakyReLU(0.2),
            )

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u4 = self.up4(d5, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.up8(u7)

        return u8


class Generator1(nn.Module):
    def __init__(self, key_num=5):
        super(Generator1, self).__init__()

        self.heat_map_l = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, key_num+1, 3, padding=1)
            )

        self.heat_map_r = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, key_num+1, 3, padding=1)
            )

        self.rgb = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 2*3, 3, padding=1)
            )

        self.depth = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 2*1, 3, padding=1)
            )

        self.softmax = nn.Softmax2d()


    def forward(self, x):

        heat_l = self.softmax(self.heat_map_l(x))
        heat_r = self.softmax(self.heat_map_r(x))
        rgb = torch.tanh(self.rgb(x))
        depth = torch.tanh(self.depth(x))

        return heat_l, heat_r, rgb, depth


class Generator2(nn.Module):
    def __init__(self, key_num=5,):
        super(Generator2, self).__init__()

        self.down1 = UNetDown(8, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 1024, dropout=0.5)

        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            )

        self.heat_map_l = nn.Sequential(
            nn.Conv2d(128, 256, 4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, key_num+1, 3, padding=1)
            )
        self.heat_map_r = nn.Sequential(
            nn.Conv2d(128, 256, 4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, key_num+1, 3, padding=1)
            )

        self.softmax = nn.Softmax2d()

        self.key_num = key_num

    def forward(self, heat_l, heat_r, rgb, depth):

        heat_l = heat_l[:,:self.key_num,:,:].sum(1).unsqueeze(1)
        heat_r = heat_r[:,:self.key_num,:,:].sum(1).unsqueeze(1)

        left_input = torch.cat((rgb[:,:3,:,:],depth[:,:1,:,:]),axis=1)
        right_input = torch.cat((rgb[:,3:,:,:],depth[:,1:,:,:]),axis=1)

        rgb_l = torch.mul(left_input,heat_l)
        rgb_r = torch.mul(right_input,heat_r)

        x = torch.cat((rgb_l,rgb_r),axis=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u4 = self.up4(d5, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        u8 = self.up8(u7)

        return self.softmax(self.heat_map_l(u8)), self.softmax(self.heat_map_r(u8))


class Representation(nn.Module):
    def __init__(self, key_num=5, clutter_num = 5):
        super(Representation, self).__init__()

        self.conv = nn.Conv2d(512, 256, 1)
        self.extract_repre = nn.Sequential(
            nn.Linear(256*9*9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
            )

        self.leaky = nn.LeakyReLU(0.2)

        self.clutter_num = clutter_num
        self.key_num = key_num

    def forward(self, feature_map, uv_list):

        feature_repre = torch.FloatTensor().cuda()
        clutter_repre = torch.FloatTensor().cuda()
        

        for i in range(self.key_num):
            l_v = int(uv_list[i,0])
            l_u = int(uv_list[i,1])
            local_feature_map_left = feature_map[:,:128,l_v-4:l_v+5,l_u-4:l_u+5]

            r_v = int(uv_list[i+self.key_num,0])
            r_u = int(uv_list[i+self.key_num,1])
            local_feature_map_right = feature_map[:,128:,r_v-4:r_v+5,r_u-4:r_u+5]

            local_feature_map = torch.cat((local_feature_map_left, local_feature_map_right),axis=1)


            k = self.leaky(self.conv(local_feature_map))
            k = k.view(1,-1)
            k = self.extract_repre(k)

            feature_repre = torch.cat((feature_repre,k))

        left_uv = uv_list[:self.key_num,:]
        right_uv = uv_list[self.key_num:,:]


        discri_l = False
        discri_r = False

        for j in range(self.clutter_num):
            H, W = feature_map.shape[-2:]
            while discri_l == False:
                c_v = random.randint(5,H-5)
                c_u = random.randint(5,W-5)
                left_uv[:,0] -= c_v
                left_uv[:,1] -= c_u
                dists = torch.norm(left_uv, dim=1)
                dists -= 10
                dists = dists > 0
                discri_l = dists.all()

            clutter_feature_map_l = feature_map[:,:128,c_v-4:c_v+5, c_u-4:c_u+5]

            while discri_r == False:
                c_v = random.randint(5,H-5)
                c_u = random.randint(5,W-5)
                right_uv[:,0] -= c_v
                right_uv[:,1] -= c_u
                dists = torch.norm(right_uv, dim=1)
                dists -= 10
                dists = dists > 0
                discri_r = dists.all()

            clutter_feature_map_r = feature_map[:,128:,c_v-4:c_v+5, c_u-4:c_u+5]

            clutter_feature_map = torch.cat((clutter_feature_map_l, clutter_feature_map_r),axis=1)

            c = self.leaky(self.conv(clutter_feature_map))
            c = c.view(1,-1)
            c = self.extract_repre(c)
            clutter_repre = torch.cat((clutter_repre,c))


        return feature_repre, clutter_repre


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, key_num =5, dex = 'can') :
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        if dex == 'heat':
            in_channels = 3+key_num
        elif dex == 'dep':
            in_channels = 3+1
        elif dex == 'rgb':
            in_channels = 3+3

        self.model = nn.Sequential(
            *discriminator_block(in_channels , 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            # *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
