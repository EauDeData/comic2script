import torch
import urllib
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



class PredictionLoss(torch.nn.Module):
    def __init__(self, model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)) -> None:
        super(PredictionLoss, self).__init__()
        self.softmax = torch.nn.Softmax(1)
        self.epsilon = 1e-5 
        self.model = model.eval()
        self.w = [.5, .5]

    def forward(self, masks, original):

        with torch.no_grad():

            output = self.model(original)['out']
            output = self.softmax(output) # Now its a probability distribution for each pixel (21 depth) is 0-1
        
        
        output  = output * masks + self.epsilon # Now there's two problems: TODO: Take a look to this: HOW WE DO TAKE ACCOUNT OF THE SELECTED ELEMENTS BEING RELEVANT TO HTE CLASSIFIER
        # 1. If mask = 0; entropy = 0, minimize by not doing anything.
        # 2. How do we compute entropy ONLY in masked elements.
        output = F.adaptive_avg_pool2d(output, (1, 1)).view(original.shape[0], -1) # Now we get if a certain category is a 0 or a 1 (average)


        entropy_loss = (-torch.sum(torch.mul(torch.log2(output), output), dim = 1)).mean() # The entropy of each category distribution
        count_loss = 1 - masks.mean() # Now maximize the count (ammount of selecteed elements)

        return count_loss * self.w[0] + entropy_loss * self.w[1]

'''

Another approach:

    We want all pixels in predicted mask to be same category.
    Therefore mask*argmax() (main category) gives you which category should be.
    mask * categories (predicted); 
    first one acts as GT, second one as H.
    Now you regularize not all being 0, in which case GT = H always since 0=0.+


'''

if __name__ == '__main__':

    l = PredictionLoss()
    bs = 5
    print(l(torch.ones(bs, 1, 64, 64, requires_grad = True), torch.ones(bs, 3, 64, 64)))