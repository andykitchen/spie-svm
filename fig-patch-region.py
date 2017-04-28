import SimpleITK as sitk

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import matplotlib.patches as patches

reader = sitk.ImageSeriesReader()

files = reader.GetGDCMSeriesFileNames('data/DOI/ProstateX-0002/1.3.6.1.4.1.14519.5.2.1.7311.5101.170110802438282744747565588784/1.3.6.1.4.1.14519.5.2.1.7311.5101.479812804428819225709948044920/')

reader.SetFileNames(files)

image = reader.Execute()

nd = sitk.GetArrayFromImage(image)

im = nd[12]
fig, ax = plt.subplots(figsize=[4, 4], dpi=300)

extent = (0, im.shape[0], -im.shape[1], 0)

ix = ax.imshow(im, extent=extent, cmap=plt.cm.gray, origin='upper')

axins = zoomed_inset_axes(ax, zoom=4, loc=1)
axins.imshow(im, extent=extent, cmap=plt.cm.gray, interpolation='nearest')

p = (160, -217)
k = 20
x1, x2, y1, y2 = p[0]-k, p[0]+k, p[1]-k, p[1]+k
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.add_patch(
    patches.Rectangle(
        (p[0] - 5, p[1] - 5),
        10,
        10,
        fill=False,
        ec='r',
        lw=2
    )
)


axins.axes.get_xaxis().set_visible(False)
axins.axes.get_yaxis().set_visible(False)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='k')

from matplotlib.ticker import FuncFormatter

def negate(x, pos):
    return '%d' % -x

formatter = FuncFormatter(negate)
ax.yaxis.set_major_formatter(formatter)

fig.savefig('figure-patch-region.pdf')
