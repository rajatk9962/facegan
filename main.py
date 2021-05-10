import torch                                       # imports pytorch
import matplotlib.pyplot as plt                    # plots graphs
from Layers import Generator
from Layers import Classif
from Subfunctions import get_noise
from Subfunctions import calculate_updated_noise
from Subfunctions import show_tensor_images
from Subfunctions import get_score



    
z_dim = 64
batch_size = 128
device = 'cpu'    


gen = Generator(z_dim).to(device)
gen_dict = torch.load("generator50.pth", map_location=torch.device(device))["model_state_dict"]
gen.load_state_dict(gen_dict)
gen.eval()

n_classes = 40
classifier = Classif(n_classes=n_classes).to(device)
class_dict = torch.load("classifierv6.pth", map_location=torch.device(device))["classifier"]
classifier.load_state_dict(class_dict)
classifier.eval()


opt = torch.optim.Adam(classifier.parameters(), lr=0.01)






feature_names = ["5oClockShadow", "ArchedEyebrows", "Attractive", "BagsUnderEyes", "Bald", "Bangs",
"BigLips", "BigNose", "BlackHair", "BlondHair", "Blurry", "BrownHair", "BushyEyebrows", "Chubby",
"DoubleChin", "Eyeglasses", "Goatee", "GrayHair", "HeavyMakeup", "HighCheekbones", "Male", 
"MouthSlightlyOpen", "Mustache", "NarrowEyes", "NoBeard", "OvalFace", "PaleSkin", "PointyNose", 
"RecedingHairline", "RosyCheeks", "Sideburn", "Smiling", "StraightHair", "WavyHair", "WearingEarrings", 
"WearingHat", "WearingLipstick", "WearingNecklace", "WearingNecktie", "Young"]
   


n_images=25




### Choose feature
def changefeature(count,feature,noise):
    fake_image_history = []
    
    
    original_classifications = classifier(gen(noise)).detach()
    
    target_indices = feature_names.index(feature) 
    other_indices = [cur_idx != target_indices for cur_idx, _ in enumerate(feature_names)]
    for i in range(count):
        opt.zero_grad()
        fake = gen(noise)
        fake_image_history += [noise]
        fake_score = get_score(
            classifier(fake), 
            original_classifications,
            target_indices,
            other_indices,
            penalty_weight=0.1
        )
        fake_score.backward()
        noise.data = calculate_updated_noise(noise, 1 / count)
    fname=show_tensor_images(gen(fake_image_history[-1]), num_images=n_images, nrow=n_images)    
    return noise,fake_image_history,fname

def genimage(count,noise,feature,fake_image_history):
    
    
    original_classifications = classifier(gen(noise)).detach()
    
    target_indices = feature_names.index(feature) 
    other_indices = [cur_idx != target_indices for cur_idx, _ in enumerate(feature_names)]    
    opt.zero_grad()
    fake = gen(noise)
    fake_image_history += [noise]
    fake_score = get_score(
        classifier(fake), 
        original_classifications,
        target_indices,
        other_indices,
        penalty_weight=0.1
    )
    fake_score.backward()
    noise.data = calculate_updated_noise(noise, 1 / count)
    
    plt.rcParams['figure.figsize'] = [n_images * 2, count * 2]
    fname=show_tensor_images(gen(fake_image_history[-1]), num_images=n_images, nrow=n_images)
    return noise,fake_image_history,fname

def genrandom():
    fake_image_history = []
    noise = get_noise(25, 64).to('cpu').requires_grad_()
    fake = gen(noise)
    fake_image_history += [noise]
    fname=show_tensor_images(fake, num_images=25, nrow=25)
    return noise,fake_image_history,fname    

def histimage(fake_image_history):
    fake_image_history.pop()
    fname=show_tensor_images(gen(fake_image_history[-1]), num_images=n_images, nrow=n_images)
    return fake_image_history,fname,fake_image_history[-1]