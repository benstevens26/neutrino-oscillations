import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

print(mpl.rcParams['figure.figsize'])
fig1 = plt.figure(figsize=(12, 8))


def plotter(figname:str = None):
    pass


