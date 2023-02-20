import math
import torch
import numpy as np


class GSP:
    def __init__(self, n_f=512, n_s=128, device='cpu'):
        self.device = torch.device(device)
        self.n_f = n_f
        self.n_s = n_s

        # 拉普拉斯矩阵
        self.init_Laplace_matrix()

        # GFT 变换矩阵
        self.init_U_matrix()

    def init_Laplace_matrix(self):
        self.L = np.zeros((self.n_f, self.n_f), np.int32)
        # L = D - A
        for i in range(self.n_f):
            for j in range(self.n_f):
                self.L[i, j] = - abs(i - j) # A_ij = |i - j|
            self.L[i, i] = np.abs(np.sum(self.L[i, :])) # 符合定义的D
            # self.L[i, i] = self.n_f - 1 # 论文中的D

    def init_U_matrix(self):
        self.U = np.zeros((self.n_f, self.n_f))
        # 特征分解
        evalues, evectors = np.linalg.eig(self.L)
        # 正交化
        for i in range(len(evalues)):
            self.U[i] = evectors[i]
            for j in range(0, i):
                self.U[i] -= np.dot(evectors[i], self.U[j]) / np.dot(self.U[j], self.U[j]) * self.U[j]
        # 归一化
        for i in range(len(evalues)):
            self.U[i] =  self.U[i] / np.sqrt(np.dot(self.U[i],  self.U[i]))

        self.U = torch.from_numpy(self.U).type(torch.FloatTensor).to(self.device)

    def gft(self, f):
        # F = U * f
        F = self.U @ f.T
        return F

    def igtf(self, F):
        # f = U_t * f
        f = self.U.T @ F.T
        return f

    def stgft(self, input_f):
        """
        Short-time Graph Fourier transform by numpy
        :return: (frames, n_f)
        """
        max_l = len(input_f)
        frames = math.ceil((max_l - self.n_f)/ self.n_s) + 1
        output_F = torch.zeros((frames, self.n_f)).to(self.device)

        # 窗厂n_f， 每次移动n_s
        for i in range(frames):

            if (i * self.n_s + self.n_f) < max_l:
                output_F[i][:] += input_f[i * self.n_s: i * self.n_s + self.n_f]
            else:
                output_F[i][: max_l - i * self.n_s] += input_f[i * self.n_s: max_l]

        # 逐帧GFT
        for i in range(frames):
            output_F[i] = self.gft(output_F[i])

        return output_F

    def istgft(self, input_F):
        """
        Inverse Short-time Graph Fourier transform by torch
        :param input_F: (frames, n_f)
        :return: n_f + (frames - 1) * n_s
        """
        # input_F: frames * self.n_f
        frames = len(input_F)
        output_f = torch.zeros(self.n_f + (frames - 1) * self.n_s).to(self.device)

        # 逐帧IGFT
        for i in range(frames):
            output_f[i * self.n_s: i * self.n_s + self.n_f] += self.igtf(input_F[i])
            # 由于是矩形窗，故重叠部分除以2
            if i != 0:
                output_f[i * self.n_s:  i * self.n_s + (self.n_f - self.n_s)] /= 2

        return output_f

    def ST_GFT(self, input_f):
        """
        :param input_f: (batch_size, n_f + (frames - 1) * n_s)
        :return: (batch_size, frames, n_f)
        """
        frames = math.ceil((input_f.shape[1] - self.n_f) / self.n_s + 1)
        output_F = torch.zeros((input_f.shape[0] , frames, self.n_f)).to(self.device)

        for i in range(input_f.shape[0]):
            output_F[i] = self.stgft(input_f[i])

        return output_F

    def iST_GFT(self, input_F):
        """
        :param input_F: (batch_size, frames, n_f)
        :return: (batch_size, n_f + (frames - 1) * n_s)
        """
        frames = input_F.shape[1]
        output_f = torch.zeros((input_F.shape[0] ,self.n_f + (frames - 1) * self.n_s)).to(self.device)

        for i in range(input_F.shape[0]):
            output_f[i] = self.istgft(input_F[i])

        return output_f

if __name__ == '__main__':
    gsp = GSP(4, 2)
    print(gsp.L)

    a = torch.Tensor([[1,2,3,4,5,6,7,8,9,10,11], [11,12,13,14,15,16,17,18,19,20,21]])
    a_F = gsp.ST_GFT(a)
    a_f = gsp.iST_GFT(a_F)
    print(a_F)
    print(a_f)

    print()


