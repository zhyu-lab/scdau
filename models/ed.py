import torch
import torch.nn as nn
import torch.nn.functional as F

class NetBlock(nn.Module):
    def __init__(
            self,
            nlayer: int,
            dim_list: list,
            act_list: list,
            dropout_rate: float,
            noise_rate: float
    ):
        """
        multiple layers netblock with specific layer counts, dimension, activations and dropout.

        Parameters
        ----------
        nlayer
            layer counts.

        dim_list
            dimension list, length equal to nlayer + 1.

        act_list
            activation list, length equal to nlayer + 1.

        dropout_rate
            rate of dropout.

        noise_rate
            rate of set part of input data to 0.

        """

        super(NetBlock, self).__init__()

        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()


        for i in range(nlayer):

            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])

            if not i == nlayer - 1:
                self.dropout_list.append(nn.Dropout(dropout_rate))

    def forward(self, x):

        x = self.noise_dropout(x)


        for i in range(self.nlayer):

            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)


            if not i == self.nlayer - 1:
                x = self.dropout_list[i](x)

        return x


class Split_Chrom_Encoder_block(nn.Module):
    def __init__(
            self,
            nlayer: int,
            dim_list: list,
            act_list: list,
            chrom_list: list,
            dropout_rate: float,
            noise_rate: float
    ):
        """
        ATAC encoder netblock with specific layer counts, dimension, activations and dropout.

        Parameters
        ----------
        nlayer
            layer counts.

        dim_list
            dimension list, length equal to nlayer + 1.

        act_list
            activation list, length equal to nlayer + 1.

        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[0].

        dropout_rate
            rate of dropout.

        noise_rate
            rate of set part of input data to 0.

        """
        super(Split_Chrom_Encoder_block, self).__init__()
        self.nlayer = nlayer
        self.chrom_list = chrom_list
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()


        for i in range(nlayer):
            if i == 0:

                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())


                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(chrom_list[j], dim_list[i + 1] // len(chrom_list)))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(dim_list[i + 1] // len(chrom_list)))
                    self.activation_list[i].append(act_list[i])
                    self.dropout_list[i].append(nn.Dropout(dropout_rate))
            else:

                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])


                if not i == nlayer - 1:
                    self.dropout_list.append(nn.Dropout(dropout_rate))

    def forward(self, x):



        x = self.noise_dropout(x)


        for i in range(self.nlayer):

            if i == 0:

                x = torch.split(x, self.chrom_list, dim=1)
                temp = []

                for j in range(len(self.chrom_list)):
                    temp.append(self.dropout_list[0][j](
                        self.activation_list[0][j](self.bn_list[0][j](self.linear_list[0][j](x[j])))))

                x = torch.concat(temp, dim=1)
            else:

                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)

                if not i == self.nlayer - 1:
                    x = self.dropout_list[i](x)

        return x


class Split_Chrom_Decoder_block(nn.Module):
    def __init__(
            self,
            nlayer: int,
            dim_list: list,
            act_list: list,
            chrom_list: list,
            dropout_rate: float,
            noise_rate: float
    ):
        """
        ATAC decoder netblock with specific layer counts, dimension, activations and dropout.

        Parameters
        ----------
        nlayer
            layer counts.

        dim_list
            dimension list, length equal to nlayer + 1.

        act_list
            activation list, length equal to nlayer + 1.

        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[end].

        dropout_rate
            rate of dropout.

        noise_rate
            rate of set part of input data to 0.

        """
        super(Split_Chrom_Decoder_block, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.chrom_list = chrom_list
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()

        for i in range(nlayer):
            if not i == nlayer - 1:
                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])
                self.dropout_list.append(nn.Dropout(dropout_rate))
            else:
                """last layer seperately forward for each chrom"""
                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())
                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(dim_list[i] // len(chrom_list), chrom_list[j]))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(chrom_list[j]))
                    self.activation_list[i].append(act_list[i])

    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            if not i == self.nlayer - 1:
                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)
                x = self.dropout_list[i](x)
            else:
                x = torch.chunk(x, len(self.chrom_list), dim=1)
                temp = []
                for j in range(len(self.chrom_list)):
                    temp.append(self.activation_list[i][j](self.bn_list[i][j](self.linear_list[i][j](x[j]))))
                x = torch.concat(temp, dim=1)

        return x



class AE_RNA(nn.Module):




    def __init__(self, RNA_input_dim, latent_size=256):
        super(AE_RNA, self).__init__()


        self.encoder = NetBlock(
            nlayer=2,
            dim_list=[RNA_input_dim, 512, latent_size],
            act_list=[nn.LeakyReLU(), nn.Identity()],
            dropout_rate=0.1,
            noise_rate=0.5
        )


        self.decoder = NetBlock(
            nlayer=2,
            dim_list=[latent_size, 512, RNA_input_dim],
            act_list=[nn.LeakyReLU(), nn.Identity()],
            dropout_rate=0.1,
            noise_rate=0
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y



class AE_ATAC(nn.Module):



    def __init__(self, ATAC_input_dim, chrom_list, latent_size=256):

        super(AE_ATAC, self).__init__()


        self.encoder = Split_Chrom_Encoder_block(
            nlayer=2,
            dim_list=[ATAC_input_dim, 32 * len(chrom_list), latent_size],
            act_list=[nn.LeakyReLU(), nn.Identity()],
            chrom_list=chrom_list,
            dropout_rate=0.1,
            noise_rate=0
            #noise_rate=0.2
        )


        self.decoder = Split_Chrom_Decoder_block(
            nlayer=2,
            dim_list=[latent_size, 32 * len(chrom_list), ATAC_input_dim],
            act_list=[nn.LeakyReLU(), nn.Sigmoid()],
            chrom_list=chrom_list,
            dropout_rate=0.1,
            noise_rate=0
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y


class AE_ADT(nn.Module):




    def __init__(self, ADT_input_dim, latent_size=256):
        super(AE_ADT, self).__init__()


        self.encoder = NetBlock(
            nlayer=2,
            dim_list=[ADT_input_dim, 512, latent_size],
            act_list=[nn.LeakyReLU(), nn.LeakyReLU()],
            dropout_rate=0.1,
            #noise_rate=0.2
            noise_rate=0
        )


        self.decoder = NetBlock(
            nlayer=2,
            dim_list=[latent_size, 512, ADT_input_dim],
            act_list=[nn.LeakyReLU(), nn.Identity()],
            dropout_rate=0.1,
            noise_rate=0
        )
            

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return z, y

class Encoder(nn.Module):



    def __init__(self, input_dim, emb_size=64):
        super(Encoder, self).__init__()


        self.net = NetBlock(
            nlayer=2,
            dim_list=[input_dim, 128, emb_size],
            act_list=[nn.LeakyReLU(), nn.Identity()],
            dropout_rate=0.1,
            noise_rate=0
        )

    def forward(self, x):
        x = self.net(x)
        return x



class Omics_label_Predictor(nn.Module):


    def __init__(self, emb_size):


        super(Omics_label_Predictor, self).__init__()


        self.fc1 = nn.Sequential(
            nn.Linear(emb_size, 5),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):

        x = self.fc1(x)
        y = F.softmax(self.fc2(x), dim=1)
        return y


class MLP(nn.Module):

    def __init__(self, emb_size, dropout_rate):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate)
        )
    def forward(self, z):
        z = self.mlp(z)
        return z



class TriEncoder(nn.Module):


    def __init__(self, input_dim, emb_size=64):

        super(TriEncoder, self).__init__()

        self.rna_encoder = Encoder(input_dim, emb_size)
        self.atac_encoder = Encoder(input_dim, emb_size)
        self.shared_encoder = Encoder(input_dim, emb_size)
        self.olp = Omics_label_Predictor(emb_size)
        # self.mlp = MLP(emb_size, dropout_rate=0.1)

        self.weights = nn.Sequential(
            nn.Linear(emb_size*2,emb_size*2),
            nn.LeakyReLU(),
            nn.Linear(emb_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # omics-specific information
        z_x = self.rna_encoder(x1)
        z_y = self.atac_encoder(x2)

        # omics-invariant informaiton
        z_x_a = self.shared_encoder(x1)
        z_y_a = self.shared_encoder(x2)

        # cross-omics information
        z_x0 = self.rna_encoder(x2)
        z_y0 = self.atac_encoder(x1)

        # omics-label predictor
        z_conxy = torch.cat([z_x, z_y], dim=0)
        y_pre = self.olp(z_conxy)

        # capture the consistency information
        p = self.weights(torch.cat([z_x_a, z_y_a], dim=1))
        z_xy = z_x_a + p * z_y_a

        return z_x, z_y, z_x0, z_y0, z_x_a, z_y_a,z_xy







