import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import uuid
import os
import copy

from torch.utils.data import DataLoader, sampler, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
import torchvision.models
from ..logger import Logger
from ..models import LeNet5, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, Simple
from ..utils import *

networks = {'LeNet5' : LeNet5,
			'resnet18' : ResNet18, 
			'resnet34' : ResNet34,
			'resnet50' : ResNet50,
			'resnet101' : ResNet101,
			'resnet152' : ResNet152,
			'Simple' : Simple}

class Trainer(object):
	def __init__(self, args, model_config):
		self.args = args
		
		# Initialize device
		self.device = None
		self.init_device()

		# Initialize dataset
		self.train_loader = None
		self.validation_loader = None
		self.test_loader = None
		self.test_loader = None
		self.num_classes = None
		self.in_channels = None
		self.batch_size_training = self.args.batch_size_training
		self.batch_size_validation= self.args.batch_size_validation
		self.init_dataset(self.args, model_config)

		self.training_epochs = self.args.training_epochs

		# Initialize model
		self.init_model()

		# Initialize log and checkpoint dir
		self.output_dir_path = './'
		self.init_output()

		# Initialize optimizer
		self.training_optimizer = None
		self.init_training_optim()

		# Initialize scheduler = None
		self.scheduler = None
		self.init_scheduler()

		# Initialize loss function
		self.criterion = None
		self.init_training_loss()

		# Initialize logger
		self.logger = None
		self.init_logger(self.output_dir_path)

	def init_device(self):
		if self.args.device == 'GPU' and torch.cuda.is_available():
			print('Using GPU device')
			self.device = 'cuda'
		else:
			print('Using CPU device')
			self.device = 'cpu'

	def init_dataset(self, args, config):
		if args.dataset == 'CIFAR10':
			normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

			builder = CIFAR10
			self.num_classes = 10
			self.in_channels = 3

			transformations = transforms.Compose([
				transforms.RandomCrop(32, padding = 4),
				transforms.ToTensor(),
			])
		elif args.dataset == 'MNIST':
			normalize = transforms.Normalize(mean = (0.1307, ), std = (0.3081, ))

			builder = MNIST
			self.num_classes = 10
			self.in_channels = 1
		
			transformations = transforms.Compose([
				transforms.Resize(28),
				transforms.CenterCrop(28),
				transforms.ToTensor(),
			])

		train_set = builder(root=args.datadir,
							train=True,
							download=True,
							transform=transformations)
		
		test_set = builder(root=args.datadir,
						   train=False,
						   download=True,
						   transform=transformations)
		
		train_set, val_set = random_split(train_set, [1 - args.validation_split, args.validation_split])


		self.train_loader = DataLoader(train_set,
									   batch_size = self.batch_size_training,
									   num_workers = self.args.num_workers,
									   shuffle = True)
		
		self.val_loader = DataLoader(val_set,
									 batch_size = self.batch_size_validation,
									 num_workers = self.args.num_workers,
									 shuffle = True)
		
		self.test_loader = DataLoader(test_set,
									  batch_size = self.batch_size_validation,
									  num_workers = self.args.num_workers)
		
	def init_model(self):
		self.best_val_acc = 0.0

		builder = networks[self.args.model_name]
		self.model = builder(num_classes = self.num_classes, in_channels = self.in_channels).to(self.device)

		if self.args.resume_from is not None and not self.args.pretrained: # resume training from checkpoint
			print('=>Loading model from checkpoint at: {}'.format(self.args.resume_from))
			package = torch.load(self.args.resume_from, map_location = self.device)
			self.model.load_state_dict(package['state_dict'])
			self.best_val_acc = package['best_val_acc']
			self.model.to(self.device)
		
		if self.args.pretrained and self.args.model_path is not None:
			print('=>Loading pretrained model')
			package = torch.load(self.args.model_path, map_location = self.device)
			self.model.load_state_dict(package['state_dict'])
			self.model.to(self.device)
			
	def init_output(self):
		name = "{}_{}".format(self.args.model_name, uuid.uuid4())
		
		self.output_dir_path = os.path.join(self.args.save_dir, name)

		if self.args.resume_from: 
			# if resuming from checkpoint, the save_dir path is two paths behind
			self.output_dir_path, _ = os.path.split(self.args.resume_from)
			self.output_dir_path, _ = os.path.split(self.output_dir_path)

		try:
			os.makedirs(self.output_dir_path)
		except FileExistsError:
			pass

		self.checkpoint_dir_path = os.path.join(self.output_dir_path, 'checkpoints')

		try:
			os.makedirs(self.checkpoint_dir_path)
		except FileExistsError:
			pass

	def init_training_optim(self):

		self.training_lr = self.args.training_lr

		if self.args.optimizer == 'Adam':
			self.training_optimizer = torch.optim.Adam(self.model.parameters(), lr = self.training_lr, weight_decay = self.args.weight_decay)
		elif self.args.optimizer == 'SGD':
			self.training_optimizer = torch.optim.SGD(self.model.parameters(), lr = self.training_lr, 
													  weight_decay = self.args.weight_decay, momentum=self.args.momentum)
		
		self.starting_epoch = 0
		
		if self.args.resume_from is not None:
			# load optimizer stats from checkpoint
			package = torch.load(self.args.resume_from, map_location = self.device)
			self.training_optimizer.load_state_dict(package['optim_dict'])
			self.starting_epoch = package['epoch'] + 1
			self.best_val_acc = package['best_val_acc']

	def init_scheduler(self):
		if self.args.scheduler == 'StepLR':
			self.scheduler = torch.optim.lr_scheduler.StepLR(self.training_optimizer, step_size = self.args.step_size)
		elif self.args.scheduler == 'CosineAnnealingLR':
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.training_optimizer, T_max = self.args.training_epochs)

	def init_training_loss(self):
		if self.args.loss == 'CrossEntropy':
			self.training_criterion = nn.CrossEntropyLoss()
		elif self.args.loss == 'SqrHinge':
			self.training_criterion = nn.SqrHingeLoss()

	def init_logger(self, output_dir_path):
		self.logger = Logger(output_dir_path)
	
	def checkpoint(self, model, epoch, val_acc, name):
		name = '{}_'.format(model) + name
		path = os.path.join(self.checkpoint_dir_path, name)

		torch.save({
			'state_dict': self.model.state_dict(),
			'optim_dict': self.training_optimizer.state_dict(),
			'epoch': epoch,
			'val_acc': val_acc,
			'best_val_acc': self.best_val_acc
		}, path)

		return path
	
	def check_accuracy(self, loader, model):
		num_correct = 0
		num_samples = 0
		model.eval() # set model to evaluation mode

		with torch.no_grad():
			for x_val, y_val in loader:
				x_val = x_val.to(self.device)
				y_val = y_val.to(self.device)
				scores = model(x_val)
				_, preds = scores.max(1)
				num_correct += (preds == y_val).sum()
				num_samples += preds.size(0)
			acc = float(num_correct) / num_samples
			self.logger.log.info('Got {} / {} correct ({})'.format(num_correct, num_samples, 100 * acc))
		
		return acc

	def train_model(self):
		name = self.args.model_name
		if self.args.pretrained == True:
			print('Storing pretrained model')
			test_accuracy = self.check_accuracy(self.test_loader, self.model)
			self.checkpoint(name, -1, test_accuracy, 'best.tar')
		else:
			print('Starting training')

			num_steps = len(self.train_loader)
			for epoch in range(self.starting_epoch, self.training_epochs):
				self.model.train()
				self.training_criterion.train()

				for i, (x_train, y_train) in enumerate(self.train_loader):
					x_train = x_train.to(self.device)
					y_train = y_train.to(self.device)

					scores = self.model(x_train)
					loss = self.training_criterion(scores, y_train)

					self.training_optimizer.zero_grad()
					loss.backward()
					self.training_optimizer.step()

					#if hasattr(self.model, 'clip_weights'):
					#	self.model.clip_weights(-1, 1)

					if i % self.args.print_every == 0:
						self.logger.log.info("Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}"
							.format(epoch, self.training_epochs, i, num_steps, loss))
				
				self.scheduler.step()
				# Check validation accuracy every epoch

				print('------- Validation accuracy -------')
				val_accuracy = self.check_accuracy(self.val_loader, self.model)

				if val_accuracy > self.best_val_acc:
					# best model is stored with best.tar at the end
					self.best_val_acc = val_accuracy
					best_path = self.checkpoint(name, epoch, val_accuracy, 'best.tar')
				else:
					# model checkpoint every epoch (overrides the checkpoint of previous epoch)
					self.checkpoint(name, epoch, val_accuracy, "_checkpoint.tar")

					# keep the checkpoint every self.args.checkpoint_every epochs
					if epoch % self.args.checkpoint_every == 0:
						self.checkpoint(name, epoch, val_accuracy, 'checkpoint_' + str(epoch) + '.tar')

			print("Training Complete")
			# Testing accuracy in the testing dataset
			print('-------- Testing Accuracy -------')
			package = torch.load(best_path, map_location = self.device)
			self.model.load_state_dict(package['state_dict'])
			self.test_acc = self.check_accuracy(self.test_loader, self.model)
			self.checkpoint(name, package['epoch'], self.test_acc, 'best.tar')
			return self.test_acc, self.model, best_path
		
	