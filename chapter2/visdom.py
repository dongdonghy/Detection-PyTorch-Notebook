import torch
import visdom

vis = visdom.Visdom(env='first')
vis.text('first visdom', win='text1')
vis.text('hello PyTorch', win='text1', append=True)

for i in range(20):
    vis.line(X=torch.FloatTensor([i]), Y=torch.FloatTensor([-i**2+20*i+1]), opts={'title': 'y=-x^2+20x+1'}, win='loss', update='append')

vis.image(torch.randn(3, 256, 256), win='random_image')
