from ij import IJ
from ij.process import ImageConverter
from ij.plugin import ChannelSplitter, RGBStackMerge

imp = IJ.getImage();
imp.setActiveChannels("0111");

IJ.run(imp, "Stack to RGB", "frames keep");
n_imp = IJ.getImage();
IJ.run(n_imp, "8-bit", "");
ImageConverter.setDoScaling(True);
IJ.run(imp, "16-bit", "");
channels = ChannelSplitter.split(imp);
lchannels = channels.tolist();
lchannels.append(n_imp);
# print(lchannels);
final_image = RGBStackMerge.mergeChannels(lchannels, False);
final_image.show();
final_image.setActiveChannels("00001");
n_imp.changes = False;
imp.changes = False;
n_imp.close();
imp.close();
print("Done");