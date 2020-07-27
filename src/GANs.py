import torch
from torchvision import utils
import os
from .models.networks import Generator, Discriminator
from .utils.logger import Logger
from torch.autograd import Variable
from torch import autograd
import torch.optim as optim
import torch.nn as nn
import time as t


class WGAN_GP(object):
    def __init__(
        self,
        output_dir='./',
        batch_size=128,
        save_per_times=50,
        img_dim=128,
        channels=3,
        learning_rate=1e-4,
        b1=0.5,
            b2=0.999):
        print("WGAN_GradientPenalty init model.")
        self.G = Generator(channels)
        self.D = Discriminator(channels)
        self.C = channels
        self.output_dir = output_dir
        self.save_per_times = save_per_times
        self.img_dim = img_dim
        # Check if cuda is available
        self.check_cuda(True)

        # WGAN values from paper
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        # WGAN_gradient penalty uses ADAM
        self.d_optimizer = optim.Adam(self.D.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.g_optimizer = optim.Adam(self.G.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        self.logger = Logger(output_dir+'/logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = 10000
        self.critic_iter = 5
        self.lambda_term = 10

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def check_cuda(self, cuda_flag=False):
        print(cuda_flag)
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))
        else:
            self.cuda = False

    def train(self, train_loader):
        self.t_begin = t.time()
        self.file = open("inception_score_graph.txt", "w")

        # Now batches are callable self.data.next()
        self.data = self.get_infinite_batches(train_loader)

        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times
            # while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                images = self.data.__next__()
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue

                z = torch.rand((self.batch_size, 100, 1, 1))

                images, z = self.get_torch_variable(images), self.get_torch_variable(z)

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                d_loss_real = self.D(images)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))

                fake_images = self.G(z)
                d_loss_fake = self.D(fake_images)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(images.data, fake_images.data)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.d_optimizer.step()
                print(f'  Discriminator iteration: {d_iter}/{self.critic_iter}, ' +
                      f'loss_fake: {d_loss_fake},' +
                      f'loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
            fake_images = self.G(z)
            g_loss = self.D(fake_images)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            g_cost = -g_loss
            self.g_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % self.save_per_times == 0:
                self.save_model(g_iter)

                if not os.path.exists(self.output_dir+'/training_result_images/'):
                    os.makedirs(self.output_dir+'/training_result_images/')

                # Denormalize images and save them in grid 8x8
                z = self.get_torch_variable(torch.randn(64, 100, 1, 1))
                samples = self.G(z)
                samples = samples.mul(0.5).add(0.5)
                samples_upscaled = nn.UpsamplingBilinear2d(scale_factor=4)(samples)
                samples = samples.data.cpu()[:64]
                samples_upscaled = samples_upscaled.data.cpu()[:64]
                grid = utils.make_grid(samples)
                utils.save_image(
                    grid,
                    self.output_dir+'/training_result_images/img_generatori_iter_{}.png'.format(str(g_iter).zfill(3)))
                grid_upscaled = utils.make_grid(samples_upscaled)
                utils.save_image(
                    grid_upscaled,
                    self.output_dir+'/training_result_images/img_generatori_iter_{}_upscaled.png'.format(str(g_iter).zfill(3)))

                # Testing
                time = t.time() - self.t_begin
                # print("Real Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                # output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                # self.file.write(output)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_cost.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data
                }
                # info = {
                #    'Wasserstein distance': Wasserstein_D,
                #    'Loss D': d_loss,
                #    'Loss G': g_cost,
                #    'Loss D Real': d_loss_real,
                #    'Loss D Fake': d_loss_fake
                # }
                # for tag, value in info.items():
                #    self.logger.scalar_summary(tag, value, g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'generated_images': self.generate_img(z, self.number_of_images)
                }

                # for tag, images in info.items():
                #    self.logger.image_summary(tag, images, g_iter + 1)

                self.generate_latent_walk(10)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # self.file.close()

        # Save the trained parameters
        self.save_model(g_iter)

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = self.get_torch_variable(torch.randn(self.batch_size, 100, 1, 1))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, self.output_dir+'/dgan_model_image.png')

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size, 1, 1, 1).uniform_(0, 1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                    prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                    prob_interpolated.size()),
                                  create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, self.img_dim, self.img_dim)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, self.img_dim, self.img_dim)[:self.number_of_images])

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, self.img_dim, self.img_dim))
            else:
                generated_images.append(sample.reshape(self.img_dim, self.img_dim))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self, g_iter):
        torch.save(self.G.state_dict(), self.output_dir+f'/generator_{g_iter}.pkl')
        torch.save(self.D.state_dict(), self.output_dir+f'/discriminator_{g_iter}.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def generate_latent_walk(self, number):
        if not os.path.exists(self.output_dir+'/interpolated_images/'):
            os.makedirs(self.output_dir+'/interpolated_images/')

        number_int = 10
        # interpolate between twe noise(z1, z2).
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5)  # denormalize
            images.append(fake_im.view(self.C, self.img_dim, self.img_dim).data.cpu())

        grid = utils.make_grid(images, nrow=number_int)
        utils.save_image(grid, self.output_dir+'/interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images.")
