import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'Plot results')
parser.add_argument('--file-name', type = str, required = True, help = 'File that contains the results')
parser.add_argument('--mode', default = 'training-progress', type = str, choices = ['training-progress'], help = 'Which plot to produce, valid options are training-progress')

args = parser.parse_args()

df = pd.read_csv(args.file_name, usecols = [0, 1, 2, 3, 4], index_col=False)

plt.style.use('seaborn-v0_8-paper')

if args.mode == 'training-progress':
	plt.figure()
	plt.plot(df['# episode'], df['reward'])
	plt.title('Episode Reward vs Episode #', size = 12)
	plt.xlabel('Episode #', size = 12)
	plt.ylabel('Episode Reward', size = 12)
	plt.savefig('training-progress.png')
