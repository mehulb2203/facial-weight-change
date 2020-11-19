# Importing all the necessary packages
import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./ali_wider')
parser.add_argument('-d1','--dir1', type=str, default='./gen_wider')
parser.add_argument('-o','--out', type=str, default='./lpips_alex.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# Crawl directories
f = open(opt.out,'w')
aligned_images = sorted(os.listdir(opt.dir0))
generated_images = sorted(os.listdir(opt.dir1))

for idx, image in enumerate(aligned_images):
	if(idx==0):
		#print("\n")
		pass
	if(image[:-4] == generated_images[idx][:-4]):
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, image))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, generated_images[idx])))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0, img1)
		#print("LPIPS score (distance) for ", image[:-4], " is:\t", round(dist01, 2), "\n")
		#print('LPIPS score (distance) for %s is:\t%.2f\n'%(image[:-4], dist01))
		f.writelines('LPIPS score (distance) for %s is:\t%.2f\n'%(image[:-4], dist01))

f.close()